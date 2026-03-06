# LLM Integration Deep Audit — Sonata Voice Pipeline

**Audit Date:** February 2025  
**Focus:** Pipeline–LLM integration, realtime API opportunities, streaming efficiency, architectural recommendations

---

## Executive Summary

The Sonata pipeline uses a well-structured **LLMClient** abstraction with three backends (Claude, Gemini, Local) and several latency optimizations (speculative prefill, streaming overlap, eager sentence flush). However, the architecture is **text-centric** and does not support audio-native LLM APIs (OpenAI Realtime, Gemini Live). Connection reuse is absent for cloud backends, and the on-device LLM has untapped speculative decoding opportunities.

---

## 1. Current Architecture Analysis

### 1.1 LLM Client Abstraction

**Location:** `src/pocket_voice_pipeline.c` lines 2042–2065

```c
typedef struct {
    void           *engine;
    LLMEngineType   type;  // LLM_ENGINE_CLAUDE | GEMINI | LOCAL

    int         (*send)(void *engine, const char *user_text);
    int         (*poll)(void *engine, int timeout_ms);
    const char *(*peek_tokens)(void *engine, int *out_len);
    void        (*consume_tokens)(void *engine, int count);
    void        (*cancel)(void *engine);
    void        (*commit_turn)(void *engine, const char *user_text);
    bool        (*is_response_done)(void *engine);
    bool        (*has_error)(void *engine);
    void        (*cleanup)(void *engine);
} LLMClient;
```

**Design notes:**

- Text-only input (`send(user_text)`); no audio input.
- Streaming output via `peek_tokens` / `consume_tokens`; no direct audio output.
- `commit_turn` used for cloud history; Local LLM keeps its own context.

### 1.2 Claude SSE Client

**Location:** `src/pocket_voice_pipeline.c` lines 2332–2722

- **Transport:** libcurl `curl_multi` for non-blocking SSE
- **SSE parsing:** Line-oriented, 8KB `SSE_LINE_BUF`, `data: ` prefix stripping
- **Per-request:** New `curl_easy_init()` per `claude_send()`; no connection reuse
- **Token extraction:** cJSON parse of `content_block_delta` → append to `c->tokens` (4KB buffer)
- **Poll:** `curl_multi_perform` + `curl_multi_poll(timeout_ms)`; typically 1 ms in `STATE_STREAMING`

**Overhead sources:**

- Per-turn TCP/TLS handshake (no HTTP keep-alive)
- JSON parse per SSE `data:` line (many events per token)
- `memcpy` into token buffer; no zero-copy path

### 1.3 Gemini SSE Client

**Location:** `src/pocket_voice_pipeline.c` lines 2724–3108

- Same pattern as Claude: new CURL easy handle per send, SSE line parsing, cJSON per event
- URL: `streamGenerateContent?alt=sse`
- History: `gemini_push_history` for user/model turns

### 1.4 Local LLM (Rust)

**Location:** `src/llm/src/lib.rs`, `src/local_llm/src/lib.rs`

- Uses `pocket_llm_*` FFI from `src/llm`
- **Throughput:** ~43 tok/s (AGENTS.md); single token per `step()`
- **API:** `set_prompt(system, user)` → `step()` → `get_token()` (decoded text)
- **Context:** Keeps last 8 turns (`MAX_CONTEXT_TURNS`)
- **No streaming text append:** Prompt is fixed at `generate_start()`; no partial transcript warmup
- **Token delivery:** `get_last_token_text()` returns only the new token’s decoded text (not cumulative)

**Differences:** `src/llm` has multi-turn, `src/local_llm` is simpler (no multi-turn per comment).

### 1.5 Pipeline State Machine

**Location:** `src/pocket_voice_pipeline.c` lines 2321–2326, 4823–5375

```
STATE_LISTENING → STATE_RECORDING → STATE_PROCESSING → STATE_STREAMING → STATE_SPEAKING
       ↑                                                      |                |
       └────────────────────── Barge-in ─────────────────────┴────────────────┘
```

- **Barge-in:** In STREAMING/SPEAKING, `voice_engine_get_barge_in()` → cancel LLM, reset TTS, go to LISTENING
- **Overlap:** Speculative prefill and streaming overlap can skip PROCESSING and go straight to STREAMING
- **Tick loop:** ~250 µs sleep in STREAMING/SPEAKING, 2 ms in LISTENING

### 1.6 Token → TTS Flow (LLM→TTS Bridge)

**Location:** `src/pocket_voice_pipeline.c` lines 5150–5227

1. `llm->poll(1)` — 1 ms wait for new data
2. `llm->peek_tokens()` → `sentbuf_add(sentbuf, token_copy, copy_len)` — accumulate in sentence buffer
3. `sentbuf_has_segment()` — flush when sentence/clause or eager word threshold
4. For each segment: `emphasis_detect_quotes` → `emphasis_predict` → `ssml_parse` → `process_segment()`
5. `process_segment()` → `tts->set_text()` or `tts->set_text_ipa()` → TTS generates audio

**Latency contributors:**

- Sentence buffer flush delay (4 words eager for Sonata)
- SSML parse, emphasis, prosody, normalization per segment
- TTS `step()` calls; `adaptive_steps_per_tick` runs 4–16 steps per tick

### 1.7 Sentence Buffer & Sub-Sentence Streaming

**Location:** `src/sentence_buffer.c` lines 359–381, `src/pocket_voice_pipeline.c` 6268–6284

- **Eager flush:** `sentbuf_set_eager(sentbuf, 4)` for Sonata, 6 for others
- Flush at word boundary when `words >= eager_flush_words` even without punctuation
- **Adaptive warmup:** First 2 sentences use `min_words=3`; then 5

**Sonata append_text integration:** `src/pocket_voice_pipeline.c` 1610–1630

When `sonata_set_text()` is called while `e->active_gen && !e->done`, it uses `sonata_lm_append_text()` to add more tokens to the LM buffer instead of resetting. This enables:

- Eager 4-word flush → first chunk to TTS
- More words appended as they arrive from LLM
- ~150–300 ms latency improvement (AGENTS.md)

### 1.8 Conversation Memory

**Location:** `src/conversation_memory.c`

- JSONL file, `memory_format_context()` for prompt augmentation
- Used in PROCESSING: `mem_ctx` prepended to transcript before `llm->send()`
- Token approximation: `strlen/4`

---

## 2. Realtime LLM API Integration Opportunities

### 2.1 OpenAI Realtime API

**Documentation:** https://platform.openai.com/docs/guides/realtime-websocket

- **Protocol:** WebSocket at `wss://api.openai.com/v1/realtime?model=gpt-realtime`
- **Audio:** 24 kHz PCM input, configurable voice output (alloy, coral, etc.)
- **Turn detection:** Built-in semantic VAD
- **Duplex:** Audio in/out over same connection

**Integration approach:**

- Add `LLM_ENGINE_OPENAI_REALTIME` and a new `RealtimeLLMClient`
- Use `src/websocket.c` (RFC 6455) or a WebSocket library for WSS
- Send `input_audio_buffer.append` events (base64 PCM)
- Receive `response.audio_transcript.done` and `response.audio.delta` (base64 audio)
- **Architectural change:** Bypass STT and TTS when using realtime; Mic → Realtime API → Speaker
- **Fallback:** Optional hybrid mode: Realtime for low-latency, STT→text→LLM→TTS for Sonata voice

### 2.2 Google Gemini Live API

**Documentation:** https://ai.google.dev/gemini-api/docs/live

- **Protocol:** Stateful WebSocket (WSS)
- **Audio:** 16 kHz 16-bit PCM input, 24 kHz output
- **Duplex:** Continuous audio streaming, barge-in support
- **Models:** `gemini-live-2.5-flash`, `gemini-2.0-flash-live`

**Integration approach:**

- Add `LLM_ENGINE_GEMINI_LIVE`
- Connect to Gemini Live WebSocket
- Stream 16 kHz mic audio (Sonata already resamples to 16 kHz for STT)
- Receive audio + optional text events
- Same bypass: Mic → Gemini Live → Speaker when using Live backend

### 2.3 Anthropic Claude Streaming

Current integration uses Messages API with `stream: true`. Anthropic does not yet offer a public audio-in/audio-out realtime API comparable to OpenAI/Gemini Live. The existing SSE path is the standard option for now.

### 2.4 Architectural Changes for Audio-Native LLMs

| Component | Current (Text Pipeline)        | Audio-Native Mode                                    |
| --------- | ------------------------------ | ---------------------------------------------------- |
| Input     | Mic → STT → transcript         | Mic → (resample if needed) → LLM WebSocket           |
| LLM       | Text API (Claude/Gemini/Local) | Audio API (OpenAI Realtime, Gemini Live)             |
| Output    | LLM text → TTS → audio         | LLM audio stream → (optional post-process) → Speaker |
| Barge-in  | Cancel LLM + reset TTS         | Native (model stops on interrupt)                    |
| EOU       | Fused EOU → final transcript   | Built-in VAD or hybrid                               |

**Recommendation:** Introduce a **BackendMode** enum: `TEXT` (current) vs `AUDIO_NATIVE`. When `AUDIO_NATIVE`:

- Skip STT and TTS for the main path
- Route audio through a `RealtimeLLMClient`-style interface
- Keep post-processing (pitch, volume, spatial) as optional passthrough

---

## 3. Current LLM Streaming Efficiency

### 3.1 SSE Parsing Overhead

- **Line buffer:** 8 KB per client; line-by-line scan for `\n`
- **JSON parse:** Every `data:` line parsed with cJSON (Claude: delta walk; Gemini: nested structure)
- **Token copy:** `memcpy` into fixed 4 KB (Claude) or 64 KB (Gemini) buffer

**Optimization ideas:**

- Incremental JSON parsing (e.g., streaming parser) to avoid full parse per event
- Larger line buffer or chunked parsing to reduce small reads
- Reuse cJSON objects where possible

### 3.2 Token-by-Token vs Batched TTS Feeding

- **Current:** Tokens go to `sentbuf_add`; TTS is fed only when a segment flushes (sentence, clause, or eager word count)
- **Batching:** Natural batching via sentence buffer; no explicit token batching before TTS
- **Sonata:** Uses `append_text` so TTS can start on 4 words and receive more incrementally

### 3.3 Speculative Prefill Effectiveness

**Location:** `src/pocket_voice_pipeline.c` lines 4966–4983

- **Threshold:** `fused_prob >= 0.55f` (AGENTS.md previously said 0.70; code uses 0.55)
- **Cancel:** `fused_prob < 0.25f` if user resumes
- **Effect:** Can skip PROCESSING and go directly to STREAMING when EOU is predicted correctly
- **Streaming overlap:** At 5 words, send partial transcript to warm LLM (lines 4924–4945)

**Impact:** AGENTS.md states ~100–300 ms latency improvement when prefill is correct.

### 3.4 LLM → TTS Bridge Latency

Components:

1. **First token to segment:** Depends on sentence length; eager flush at 4 words helps
2. **Segment processing:** SSML parse, emphasis, prosody, normalization (CPU-bound)
3. **TTS first audio:** Sonata Flow + LM generation; 12 tokens first chunk (~240 ms audio)
4. **Adaptive steps:** Up to 16 TTS steps per tick when buffer is empty

AGENTS.md: End-to-end first chunk ~543–600 ms; Voice Response Latency (VRL) target < 500 ms.

### 3.5 Connection Reuse / Keep-Alive

**Finding:** No HTTP keep-alive or connection reuse.

- `claude_send()` / `gemini_send()` call `curl_easy_cleanup` on previous handle and `curl_easy_init()` for each request
- Each turn = new TCP + TLS handshake (~50–150 ms depending on RTT)

**Recommendation:** Retain `curl_easy` across turns when possible:

- Reuse handle, only change `CURLOPT_POSTFIELDS`
- Or use `curl_multi` with a persistent easy handle
- Ensure `CURLOPT_TCP_KEEPALIVE` (or equivalent) for long idle periods

---

## 4. Pipeline State Machine Optimization

### 4.1 State Transition Overhead

- **Minimal:** State is an enum; `pipeline_tick` returns `next`; no heavy work on transition
- **Barge-in:** Cancels LLM, resets TTS, drains buffers — expected for correctness

### 4.2 Overlapping States

- **Current overlap:** RECORDING can send speculative/streaming overlap; PROCESSING can be skipped
- **Further overlap:** STT could keep running during early STREAMING (e.g., for confidence); current design finalizes transcript at EOU

### 4.3 Barge-in Efficiency

- Single check at top of `pipeline_tick`
- `voice_engine_clear_barge_in()` and `barge_in_flush()` ensure clean reset
- Emotion-aware scaling of VAD threshold for empathetic content

### 4.4 Turn-Taking Latency

- `usleep(250)` in STREAMING/SPEAKING adds ~250 µs per tick
- Poll timeout 1 ms in `llm->poll(1)` can add up if no data
- Consider non-blocking poll (`timeout_ms=0`) when data is available to reduce idle time

---

## 5. Multi-Modal LLM Support

### 5.1 Sending Audio to Multimodal LLMs

**Current:** Only text is sent. Claude/Gemini text APIs do not accept audio.

**To support audio input (e.g., Gemini Multimodal):**

- Encode audio (e.g., base64) and send as a `part` with `inline_data`
- Requires segmenting or chunking audio (e.g., per utterance or fixed blocks)
- Same duplex pattern as Gemini Live but in request/response style

### 5.2 Streaming Audio Input

- **Realtime APIs:** Designed for streaming audio in and out
- **REST APIs:** Typically full request body; no native streaming of input audio
- **Hybrid:** Use Live/Realtime for low-latency voice; use REST for complex or mixed modality

### 5.3 Duplex Patterns

- **OpenAI Realtime / Gemini Live:** Single WebSocket, bidirectional audio
- **Sonata today:** Request-response per turn; no persistent audio duplex to LLM

---

## 6. On-Device LLM Optimization

### 6.1 Current Throughput

- **~43 tok/s** (AGENTS.md) for Llama-3.2-3B on Metal
- `MAX_NEW_TOKENS: 256` in `src/llm/src/lib.rs` line 29

### 6.2 KV Cache Management

- `model::Cache::new(...)` per `generate_start()`; cache is discarded at end of turn
- No explicit KV eviction or sliding window; full context used

### 6.3 Speculative Decoding

- **Sonata LM** has speculative decoding (draft model) for TTS semantic LM — see `sonata_lm_load_draft`, `sonata_lm_speculate_step`
- **On-device text LLM** (`pocket_llm`): No speculative decoding; one token per `step()`
- **Opportunity:** Add draft model (e.g., 1B) for text LLM to validate with main model and improve tok/s

### 6.4 Memory Bandwidth

- BF16 on Metal; KV cache in device memory
- Single-token step may underutilize GPU; batch or speculative decoding could improve utilization

---

## 7. Sub-Sentence Streaming

### 7.1 Current Implementation

- **Eager flush:** 4 words (Sonata), 6 (others) — `sentbuf_set_eager(sentbuf, 4)` at line 6275
- **Sonata append_text:** When a new segment arrives and TTS is already generating, `sonata_set_text` uses `append_text` to extend the LM buffer (lines 1615–1629)
- **Disable after first sentence:** `sb->eager_flush_words = 0` after first real sentence boundary (line 330)

### 7.2 Could It Be More Aggressive?

- **3 words:** Possible; risk of awkward prosody and more tiny chunks
- **2 words:** Likely too aggressive; more overhead, worse flow
- **A/B test:** Try 3 vs 4 words and measure VRL and MOS

### 7.3 Optimal Flush Threshold

- **4 words** is a reasonable default for Sonata
- **Adaptive:** `sentbuf_set_adaptive(sentbuf, 2, 3)` — first 2 sentences use 3 words, then 5
- Recommendation: Make eager threshold configurable (e.g., `--eager-words 3|4|5`) for tuning

---

## 8. Recommendations Summary

### Ranked by Latency Impact

| Priority | Recommendation                                | Impact                       | Effort |
| -------- | --------------------------------------------- | ---------------------------- | ------ |
| 1        | **HTTP connection reuse** for Claude/Gemini   | High (~50–150 ms/turn)       | Low    |
| 2        | **Non-blocking poll** when tokens available   | Medium (~1–5 ms)             | Low    |
| 3        | **Configurable eager threshold**              | Medium (tuning)              | Low    |
| 4        | **OpenAI Realtime API backend**               | Very high (bypass STT+TTS)   | High   |
| 5        | **Gemini Live API backend**                   | Very high (bypass STT+TTS)   | High   |
| 6        | **On-device LLM speculative decoding**        | Medium (higher tok/s)        | Medium |
| 7        | **Incremental/streaming JSON parser** for SSE | Low–Medium                   | Medium |
| 8        | **Hybrid audio-native mode**                  | Architecture for future APIs | Medium |

### Architectural Recommendations

1. **Introduce `LLMClient` variants for audio-native backends**
   - New ops: `send_audio()`, `peek_audio()`, or an event callback model
   - Pipeline selects TEXT vs AUDIO_NATIVE at startup

2. **WebSocket support for realtime APIs**
   - Reuse or extend `src/websocket.c` for WSS
   - Handle JSON events (base64 audio, session events)

3. **Preserve current path**
   - Keep STT→LLM→TTS for Sonata TTS and configurable voice
   - Use audio-native backends only when explicitly selected

---

## 9. Key File Reference

| File                          | Purpose                                                              |
| ----------------------------- | -------------------------------------------------------------------- |
| `src/pocket_voice_pipeline.c` | LLMClient, Claude/Gemini/Local clients, state machine, pipeline_tick |
| `src/sentence_buffer.c`       | Token accumulation, eager flush, sentence/clause boundaries          |
| `src/sentence_buffer.h`       | API for eager flush, adaptive warmup                                 |
| `src/conversation_memory.c`   | Multi-turn context, JSONL persistence                                |
| `src/llm/src/lib.rs`          | On-device Llama (pocket_llm), step, get_token                        |
| `src/local_llm/src/lib.rs`    | Alternative local LLM implementation                                 |
| `src/http_api.c`              | REST + WebSocket /v1/stream (STT→LLM→TTS)                            |
| `src/websocket.c`             | RFC 6455 WebSocket, used by HTTP API                                 |

---

_Generated by LLM integration audit — Sonata project_
