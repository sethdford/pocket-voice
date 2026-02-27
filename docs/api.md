# HTTP API Reference

Sonata includes a built-in HTTP API server for programmatic access to STT, TTS, and chat functionality. The API is OpenAI-compatible where applicable, making it a drop-in replacement for many workflows.

## Starting the Server

```bash
# Start on default port 8080
./sonata --server --server-port 8080

# With API key authentication
SONATA_API_KEY=my-secret-key ./sonata --server --server-port 8080

# With JSON config
./sonata --config config.json
```

Config file example:

```json
{
  "server": { "enabled": true, "port": 8080 }
}
```

## Authentication

If `SONATA_API_KEY` is set (or configured via `http_api_set_api_key()`), all endpoints except `/health` require a Bearer token:

```bash
curl -H "Authorization: Bearer my-secret-key" http://localhost:8080/v1/voices
```

If no API key is configured, all endpoints are open access.

## Server Architecture

- **Thread pool**: 4 worker threads + 1 accept thread
- **Rate limiting**: 60 requests/sec with burst capacity of 10
- **Max request body**: 16 MB
- **Request timeout**: 30 seconds
- **CORS**: Enabled for all origins

---

## Endpoints

### `GET /health` (alias: `/v1/health`)

Health check. No authentication required.

```bash
curl http://localhost:8080/health
```

**Response:**

```json
{ "status": "ok", "version": "pocket-voice 1.0" }
```

---

### `POST /v1/audio/transcriptions`

Transcribe audio to text. Send WAV audio in the request body.

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Content-Type: audio/wav" \
  --data-binary @audio.wav
```

**Response:**

```json
{ "text": "hello world", "samples": 48000 }
```

**With word timestamps:**

```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions?word_timestamps=true" \
  -H "Content-Type: audio/wav" \
  --data-binary @audio.wav
```

```json
{
  "text": "hello world",
  "samples": 48000,
  "word_timestamps": [
    { "word": "hello", "start": 0.0, "end": 0.45 },
    { "word": "world", "start": 0.45, "end": 0.9 }
  ]
}
```

**Supported input**: WAV (8/16/24/32-bit, mono or stereo).

---

### `POST /v1/audio/speech`

Text-to-speech synthesis. Accepts plain text or JSON body.

#### Plain Text

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -d "Hello, world!" -o output.wav
```

#### JSON (OpenAI-compatible)

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, world!",
    "voice": "alloy",
    "response_format": "opus"
  }' -o output.opus
```

#### Full JSON Options

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "voice": "0",
    "emotion": "happy",
    "speed": 1.0,
    "volume": 1.0,
    "stream": false,
    "word_timestamps": false,
    "response_format": "wav",
    "output_format": {
      "sample_rate": 24000,
      "encoding": "pcm_s16le",
      "container": "wav"
    },
    "pronunciation_overrides": [
      {"text": "GIF", "pronunciation": "jiff"}
    ]
  }' -o output.wav
```

#### Request Fields

| Field                     | Type   | Default | Description                                    |
| ------------------------- | ------ | ------- | ---------------------------------------------- |
| `text` or `input`         | string | —       | Text to synthesize (required, max 10KB)        |
| `voice`                   | string | `"0"`   | Voice ID or OpenAI voice name                  |
| `emotion`                 | string | —       | Emotion hint (happy, sad, excited, calm, etc.) |
| `speed`                   | float  | `1.0`   | Speaking rate (0.25 – 4.0)                     |
| `volume`                  | float  | `1.0`   | Volume multiplier (0.5 – 2.0)                  |
| `stream`                  | bool   | `false` | Enable chunked streaming response              |
| `word_timestamps`         | bool   | `false` | Include estimated word timing                  |
| `response_format`         | string | `"wav"` | Output format: `wav`, `opus`, `mp3`, `pcm`     |
| `pronunciation_overrides` | array  | —       | Per-request pronunciation rules                |

#### Output Format Options

Nested under `output_format`:

| Field         | Type   | Default       | Values                                            |
| ------------- | ------ | ------------- | ------------------------------------------------- |
| `sample_rate` | int    | `24000`       | 8000, 16000, 22050, 24000, 44100, 48000           |
| `encoding`    | string | `"pcm_s16le"` | `pcm_s16le`, `pcm_f32le`, `pcm_mulaw`, `pcm_alaw` |
| `container`   | string | `"wav"`       | `wav`, `raw`, `mp3`, `opus`                       |

#### OpenAI Voice Mapping

| OpenAI Name | Sonata Speaker ID |
| ----------- | ----------------- |
| alloy       | 0                 |
| echo        | 1                 |
| fable       | 2                 |
| onyx        | 3                 |
| nova        | 4                 |
| shimmer     | 5                 |

#### Streaming TTS

When `"stream": true`, the response uses HTTP chunked Transfer-Encoding. Audio chunks are sent as they are generated, reducing time-to-first-audio.

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "A long story...", "stream": true}' \
  --no-buffer -o output.wav
```

#### Word Timestamps

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "word_timestamps": true}'
```

Returns JSON with estimated word boundaries instead of audio:

```json
{
  "words": [
    { "word": "Hello", "start": 0.0, "end": 0.25 },
    { "word": "world", "start": 0.25, "end": 0.5 }
  ]
}
```

---

### `POST /v1/chat`

Send a text message and get a response from the configured LLM backend.

```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke"}'
```

---

### `POST /v1/voices`

Register a cloned voice from a WAV reference file. Requires a speaker encoder model to be loaded.

```bash
curl -X POST http://localhost:8080/v1/voices \
  -H "Content-Type: audio/wav" \
  --data-binary @reference_voice.wav
```

**Response:**

```json
{ "voice_id": "voice_0", "embedding_dim": 512 }
```

The returned `voice_id` can be used in TTS requests:

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello in a cloned voice", "voice": "voice_0"}' \
  -o output.wav
```

Up to 32 cloned voices can be stored in memory simultaneously.

---

### `GET /v1/voices`

List all registered cloned voices.

```bash
curl http://localhost:8080/v1/voices
```

---

### `GET /v1/stream`

WebSocket upgrade endpoint for real-time bidirectional audio streaming.

```bash
# Using wscat
wscat -c ws://localhost:8080/v1/stream
```

See [WebSocket Protocol](#websocket-protocol) below.

---

## WebSocket Protocol

The `/v1/stream` endpoint upgrades to a WebSocket connection (RFC 6455) for real-time voice interaction.

### Connection

```
GET /v1/stream HTTP/1.1
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: <key>
```

### Message Types

#### Client → Server

| Type   | Format        | Description                      |
| ------ | ------------- | -------------------------------- |
| Binary | Raw PCM audio | Audio frames from microphone     |
| Text   | `flush`/`end` | Manually trigger STT flush       |
| Text   | JSON config   | Negotiate Opus codec (see below) |

#### Server → Client

| Type        | Format                                                     | Description                       |
| ----------- | ---------------------------------------------------------- | --------------------------------- |
| Text (JSON) | `{"type":"listening"}`                                     | Pipeline is ready for audio       |
| Text (JSON) | `{"type":"transcript","text":"..."}`                       | STT transcription result          |
| Text (JSON) | `{"type":"processing"}`                                    | Transcript received, calling LLM  |
| Text (JSON) | `{"type":"llm_token","text":"..."}`                        | Streaming LLM token               |
| Text (JSON) | `{"type":"speaking"}`                                      | TTS audio playback started        |
| Text (JSON) | `{"type":"error","message":"..."}`                         | Engine unavailable or other error |
| Text (JSON) | `{"type":"config_ack","codec":"opus","sample_rate":16000}` | Opus codec negotiation confirmed  |
| Binary      | Raw PCM audio                                              | TTS audio chunks                  |

### Flow

1. Client connects and receives `{"type":"listening"}`
2. Client sends binary audio frames
3. Server detects end-of-utterance and sends `{"type":"transcript","text":"..."}`
4. Server streams LLM tokens as `{"type":"llm_token","text":"..."}`
5. Server sends `{"type":"speaking"}` followed by binary audio chunks
6. Server returns to `{"type":"listening"}` when complete

### Keep-Alive

The server responds to WebSocket PING frames with PONG automatically.

---

## Audio Encoding Reference

### Input (STT)

- **Format**: WAV (RIFF)
- **Bit depths**: 8, 16, 24, or 32-bit
- **Channels**: Mono or stereo (stereo downmixed to mono)
- **Sample rate**: Any (internally resampled to 16/24kHz)

### Output (TTS)

| Container | Content-Type               | Description                    |
| --------- | -------------------------- | ------------------------------ |
| WAV       | `audio/wav`                | Standard RIFF WAV              |
| Raw PCM   | `application/octet-stream` | Headerless raw samples         |
| Opus      | `audio/opus`               | Ogg Opus container (RFC 7845)  |
| MP3       | `audio/mpeg`               | Requires libmp3lame (optional) |

| Encoding    | Description                                    |
| ----------- | ---------------------------------------------- |
| `pcm_s16le` | 16-bit signed integer, little-endian (default) |
| `pcm_f32le` | 32-bit IEEE float, little-endian               |
| `pcm_mulaw` | G.711 mu-law (8-bit, telephony)                |
| `pcm_alaw`  | G.711 A-law (8-bit, telephony)                 |

---

## Error Responses

All errors return JSON:

```json
{ "error": "description of the error" }
```

| Status | Meaning                                             |
| ------ | --------------------------------------------------- |
| 400    | Bad request (missing body, invalid WAV, empty text) |
| 401    | Unauthorized (missing or invalid API key)           |
| 413    | Payload too large (text > 10KB or body > 16MB)      |
| 429    | Rate limited (exceeds 60 req/s)                     |
| 500    | Internal server error (TTS/STT engine failure)      |
