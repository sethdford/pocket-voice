# Red-Team Security Audit — Sonata Wave 1 Upgrades

## CRITICAL FINDINGS (P0 — Immediate Crash/Exploit)

### P0-1: FSQ Index Arithmetic Overflow Before Bounds Check

**File:** `src/codec_12hz.c`, Lines 177-178
**Function:** `fsq_dequantize()`

```c
int idx = (indices[0] * 512) + (indices[1] * 64) + (indices[2] * 8) + indices[3];
if (idx >= codec->cfg.fsq_codebook_size) idx = 0;  /* Bounds check */
```

**Vulnerability:**

- FSQ indices are supposed to be in range [0,7] (4-bit each, 8^4=4096 codebook)
- Bounds validation happens in `codec_12hz_decode_frame()` at lines 668-671, BEFORE calling `fsq_dequantize()`
- However, the arithmetic can overflow: `(255 * 512) = 130,560` exceeds int32 max on some platforms
- Result is clamped to 0 silently if out of bounds — no error return

**Proof of Concept:**

```c
uint8_t bad_indices[4] = {255, 255, 255, 255};  // Max values
float acoustic_latent[512] = {0};
int n = codec_12hz_decode_frame(codec, bad_indices, acoustic_latent, out_audio);
// idx = 255*512 + 255*64 + 255*8 + 255 = 147,711
// Reads out-of-bounds codebook entry or wraps to garbage index
// Silent corruption, difficult to debug
```

**Attack Impact:**

- Out-of-bounds read from heap (codebook)
- Information disclosure (reads adjacent memory)
- Silent audio corruption (no error returned)

**Fix:**

```c
// In codec_12hz_decode_frame(), validate BEFORE calling fsq_dequantize():
for (int i = 0; i < fsq_dim; i++) {
    if (semantic_codes[i] > 7) {
        fprintf(stderr, "[codec_12hz] FSQ index out of range: %u\n", semantic_codes[i]);
        return 0;
    }
}
// Now safe to compute idx
fsq_dequantize(codec, semantic_codes, semantic_embedding);
```

**Severity:** **P0** (Out-of-bounds read, information disclosure)

---

### P0-2: Integer Overflow in Input Projection Allocation

**File:** `src/codec_12hz.c`, Lines 253-258
**Function:** `codec_12hz_create_empty()`

```c
int input_dim = (cfg->fsq_dim * cfg->fsq_embed_dim) + cfg->acoustic_dim;
codec->input_proj_weight = (float *)calloc(
    (size_t)cfg->dec_dim * input_dim * 7,
    sizeof(float)
);
```

**Vulnerability:**
Three unbounded multiplications of user-controlled config values:

- If `cfg->dec_dim = 0x80000000` (large power of 2)
- `input_dim = cfg->fsq_dim * 512 + 512 ≈ 2600`
- Total allocation: `(uint64_t)0x80000000 * 2600 * 7` → **overflow**
- `calloc()` returns NULL on overflow (implementation-dependent)
- No NULL check after allocation — subsequent dereference crashes

**Proof of Concept:**

```c
Codec12HzConfig cfg = {
    .dec_dim = 0x80000000,  // Malicious config
    .fsq_dim = 4,
    .fsq_embed_dim = 512,
    .acoustic_dim = 512,
    // ... other fields
};
Codec12Hz *codec = codec_12hz_create_empty(&cfg);
// calloc((uint64_t)0x80000000 * 2600 * 7, sizeof(float)) → size_t overflow
// codec->input_proj_weight == NULL
// decode_frame() dereferences NULL → SIGSEGV
```

**Attack Impact:**

- NULL pointer dereference → crash
- Memory exhaustion (failed allocations treated as success)

**Fix:**

```c
// Validate config ranges early:
if (cfg->dec_dim > 8192 || cfg->fsq_dim > 256 || cfg->fsq_embed_dim > 2048) {
    return NULL;  // Reject unreasonable configs
}
// Safe arithmetic with overflow checks
if (cfg->fsq_dim > UINT32_MAX / cfg->fsq_embed_dim) return NULL;
int input_dim = (cfg->fsq_dim * cfg->fsq_embed_dim) + cfg->acoustic_dim;
if (cfg->dec_dim > UINT64_MAX / (input_dim * 7)) return NULL;

codec->input_proj_weight = (float *)calloc(
    (size_t)cfg->dec_dim * input_dim * 7,
    sizeof(float)
);
if (!codec->input_proj_weight) {
    codec_12hz_destroy(codec);
    return NULL;  // Fail gracefully
}
```

**Severity:** **P0** (NULL dereference, memory exhaustion)

---

### P0-3: Heap Buffer Overflow in Output Projection Loop

**File:** `src/codec_12hz.c`, Lines 764-768
**Function:** `codec_12hz_decode_frame()`

```c
for (int c = 0; c < dec_dim; c++) {
    int w_idx = ((t + c) * 7) / hop_length;
    if (w_idx >= 7) w_idx = 6;
    if (c * 7 + w_idx < (size_t)1 * dec_dim * 7) {
        sum += codec->output_weight[c * 7 + w_idx] * x[x_idx];
    }
}
```

**Vulnerability:**

- `output_weight` allocated at line 323: `[1 * (cfg->dec_dim / 16) * 7]`
  - With `dec_dim = 768`: **336 floats total** (48 \* 7)
- Loop iterates `c` from 0 to `dec_dim-1` (0 to 767)
- Index calculation: `c * 7 + w_idx`
  - When `c=100`: `100 * 7 = 700` (**exceeds 336!**)
- Bounds check at line 766: `if (c * 7 + w_idx < (size_t)1 * dec_dim * 7)`
  - Check is: `700 < 768 * 7 = 5376` → **TRUE** (passes!)
  - **But actual array size is `48 * 7 = 336`, not `768 * 7`**
- **Heap buffer overflow** → writes out-of-bounds

**Proof of Concept:**

```c
Codec12HzConfig cfg = { .dec_dim = 768, /* ... */ };
Codec12Hz *codec = codec_12hz_create_empty(&cfg);

uint8_t semantic_codes[4] = {0, 0, 0, 0};
float acoustic_latent[512] = {0};
float out_audio[1920] = {0};

// Decode a frame
codec_12hz_decode_frame(codec, semantic_codes, acoustic_latent, out_audio);
// In output projection loop:
//   c=100: offset = 700, but output_weight has only 336 floats
//   Writes to heap: output_weight[700] = corrupted_sum
//   Overwrites adjacent heap metadata or adjacent allocations
// Potential RCE via heap metadata corruption
```

**Attack Impact:**

- Heap buffer overflow
- Arbitrary heap memory corruption
- Potential RCE (if attacker controls heap layout)

**Fix:**

```c
// Correct bounds check — use actual allocated size
int output_weight_size = (codec->cfg.dec_dim / 16) * 7;
for (int c = 0; c < codec->cfg.dec_dim; c++) {
    int w_idx = ((t + c) * 7) / hop_length;
    if (w_idx >= 7) w_idx = 6;

    // Check against ACTUAL size
    if (c < (codec->cfg.dec_dim / 16) && c * 7 + w_idx < output_weight_size) {
        sum += codec->output_weight[c * 7 + w_idx] * x[x_idx];
    }
}
```

**Severity:** **P0** (Heap buffer overflow, RCE potential)

---

### P0-4: Path Traversal in Speaker Encoder Config Loading

**File:** `src/speaker_encoder.c`, Lines 50-60
**Function:** `speaker_encoder_create()`

```c
char config_path[1024] = {0};
const char *last_slash = strrchr(weights_path, '/');
if (last_slash) {
    int dir_len = last_slash - weights_path;
    snprintf(config_path, sizeof(config_path) - 1,
             "%.*s/speaker_encoder_config.json", dir_len, weights_path);
} else {
    snprintf(config_path, sizeof(config_path) - 1, "speaker_encoder_config.json");
}
```

**Vulnerability:**

- No canonicalization of constructed path
- Directory traversal via `../` sequences
- No validation that config_path is within expected directory

**Proof of Concept:**

```c
// Attacker provides malicious weights_path:
const char *weights_path = "/tmp/weights/../../etc/shadow/speaker_encoder.safetensors";

speaker_encoder_create(weights_path);
// Extracts directory: "/tmp/weights/../../etc/shadow"
// Constructs config_path: "/tmp/weights/../../etc/shadow/speaker_encoder_config.json"
// Loads sensitive file by traversing directories

// Or worse:
weights_path = "/tmp/weights/../../../../tmp/attacker_config.json/speaker_encoder.safetensors";
// config_path = "/tmp/weights/../../../../tmp/attacker_config.json/speaker_encoder_config.json"
// Loads attacker-controlled config file instead of legitimate one
```

**Attack Impact:**

- Arbitrary file read (e.g., `/etc/passwd`)
- Load attacker-controlled config files
- Potentially inject malicious paths

**Fix:**

```c
#include <stdlib.h>
#include <limits.h>

char real_weights[PATH_MAX];
if (!realpath(weights_path, real_weights)) {
    fprintf(stderr, "[speaker_encoder] Invalid path: %s\n", weights_path);
    return NULL;
}

// Now extract directory safely from canonicalized path
const char *last_slash = strrchr(real_weights, '/');
if (last_slash) {
    int dir_len = last_slash - real_weights;
    snprintf(config_path, sizeof(config_path) - 1,
             "%.*s/speaker_encoder_config.json", dir_len, real_weights);
} else {
    snprintf(config_path, sizeof(config_path) - 1, "speaker_encoder_config.json");
}

// Verify config_path is under expected directory (optional extra defense)
char real_config[PATH_MAX];
if (realpath(config_path, real_config)) {
    // Check that real_config starts with expected directory
    // if (strncmp(real_config, expected_dir, strlen(expected_dir)) != 0) {
    //     return NULL;  // Config outside expected directory
    // }
}
```

**Severity:** **P0** (Arbitrary file read via path traversal)

---

### P0-5: Unbounded File Read in Metadata Parsing (OOM)

**File:** `src/sonata_flow/src/lib.rs`, Lines 3686-3692
**Function:** `is_distilled_checkpoint()`

```rust
fn is_distilled_checkpoint(weights_path: &str) -> bool {
    match std::fs::read(weights_path) {
        Ok(data) => {
            match SafeTensors::read_metadata(&data) {
                Ok((_header_size, metadata)) => {
                    if let Some(meta_map) = metadata.metadata() {
                        if let Some(val) = meta_map.get("distilled") {
                            return val.to_lowercase() == "true";
                        }
                    }
                    false
                }
                Err(_) => false,
            }
        }
        Err(_) => false,
    }
}
```

**Vulnerability:**

- `std::fs::read()` loads **entire file into memory** without size limit
- Attacker-controlled file path (e.g., from config or user input)
- No early check for file size

**Proof of Concept:**

```bash
# Create a 5GB malicious safetensors file
dd if=/dev/zero of=/tmp/5gb_weights.safetensors bs=1G count=5

# In Rust code:
let weights_path = "/tmp/5gb_weights.safetensors";
is_distilled_checkpoint(weights_path);
// std::fs::read() tries to allocate 5GB
// OOM killer terminates program → DoS
```

**Attack Impact:**

- Out-of-memory crash
- Denial of service (program crash via unbounded allocation)

**Fix:**

```rust
fn is_distilled_checkpoint(weights_path: &str) -> bool {
    const MAX_FILE_SIZE: u64 = 1_000_000_000;  // 1GB limit

    // Check file size before reading
    match std::fs::metadata(weights_path) {
        Ok(metadata) => {
            if metadata.len() > MAX_FILE_SIZE {
                eprintln!("[is_distilled] File too large: {} bytes", metadata.len());
                return false;  // Assume non-distilled, skip check
            }
        }
        Err(_) => return false,
    }

    // Safe to read now
    match std::fs::read(weights_path) {
        Ok(data) => {
            match SafeTensors::read_metadata(&data) {
                Ok((_header_size, metadata)) => {
                    if let Some(meta_map) = metadata.metadata() {
                        if let Some(val) = meta_map.get("distilled") {
                            return val.to_lowercase() == "true";
                        }
                    }
                    false
                }
                Err(_) => false,
            }
        }
        Err(_) => false,
    }
}
```

**Severity:** **P0** (Denial of service via memory exhaustion)

---

## HIGH-PRIORITY FINDINGS (P1 — Crash/Data Corruption)

### P1-1: Ring Buffer Wraparound Logic Error in Streaming Mode

**File:** `src/codec_12hz.c`, Lines 783-810
**Function:** `codec_12hz_decode_frame()` (streaming mode)

```c
int head = codec->ring_head;
int first = ring_n - head;  // Remaining space from head to end of ring

if (first >= ring_n) {  // This condition is IMPOSSIBLE if head in [0, ring_n)
    /* No wrap — add directly */
    vDSP_vadd(codec->overlap_buf, 1, audio_frame, 1,
              codec->overlap_buf, 1, hop_length);
} else {
    /* Two-part add (wraparound) */
    vDSP_vadd(codec->overlap_buf + head, 1, audio_frame, 1,
              codec->overlap_buf + head, 1, first);
    vDSP_vadd(codec->overlap_buf, 1, audio_frame + first, 1,
              codec->overlap_buf, 1, hop_length - first);
}
```

**Vulnerability:**

- Condition `if (first >= ring_n)` is impossible:
  - `first = ring_n - head`
  - If `head ∈ [0, ring_n)`, then `first ∈ (0, ring_n]`
  - `first >= ring_n` only true if `head ≤ 0`, but head is always ≥ 0
- Loop always takes "wraparound" path, even when no wrap needed
- Results in writing beyond intended buffer boundary

**Proof of Concept:**

```c
ring_n = 4096, hop_length = 1920, head = 500
first = 4096 - 500 = 3596

Condition: if (3596 >= 4096) → FALSE
→ Takes "Two-part add" path (wraparound)

But first (3596) > hop_length (1920), so NO wraparound needed!
vDSP_vadd(overlap_buf + 500, ..., ..., 3596);  // Writes 3596 floats
  Should write: 1920 floats

Result: Writes 3596 samples instead of 1920
→ Corrupts overlap buffer, overwriting next frame's data
→ Audio glitches, clicks, or crash
```

**Attack Impact:**

- Audio buffer corruption
- Audio dropout / glitches
- Potential crash if heap metadata corrupted

**Fix:**

```c
int head = codec->ring_head;

// Correct condition: can we fit hop_length samples without wrapping?
if (head + hop_length <= ring_n) {
    /* No wrap — add directly */
    vDSP_vadd(codec->overlap_buf + head, 1, audio_frame, 1,
              codec->overlap_buf + head, 1, hop_length);
} else {
    /* Wraparound needed */
    int first_part = ring_n - head;  // Samples until end of buffer
    int second_part = hop_length - first_part;  // Samples wrapping to beginning

    vDSP_vadd(codec->overlap_buf + head, 1, audio_frame, 1,
              codec->overlap_buf + head, 1, first_part);
    vDSP_vadd(codec->overlap_buf, 1, audio_frame + first_part, 1,
              codec->overlap_buf, 1, second_part);
}

codec->ring_head = (head + hop_length) % ring_n;
```

**Severity:** **P1** (Buffer corruption, audio glitches, crash)

---

### P1-2: No Validation of LM Hidden State Shape in Drafter

**File:** `src/sonata_lm/src/drafter.rs`, Lines 124-171
**Function:** `GruDrafter::draft()`

```rust
pub fn draft(&self, lm_hidden: &Tensor, current_token: u32, num_steps: usize) -> Result<Vec<u32>> {
    let h0_squeezed = if lm_hidden.dims().len() == 3 {
        lm_hidden.squeeze(1)?  // Assumes dim 1 is size 1!
    } else {
        lm_hidden.clone()
    };

    // Project LM hidden state to GRU initial state
    let h0 = self.hidden_proj.forward(&h0_squeezed)?;  // No shape validation!

    // ... rest of draft loop
}
```

**Vulnerability:**

- No validation that `h0_squeezed` has shape `(batch, d_model)`
- If caller passes wrong dimension size, `hidden_proj` produces garbage
- No error checking on dimension mismatch

**Proof of Concept:**

```rust
DrafterConfig { d_model: 512, ... }

// Attacker (or bug) passes wrong shape:
let lm_hidden = Tensor::randn(0.0, 1.0, (1, 256), &device)?;  // Wrong! Expected d_model=512

let draft_tokens = drafter.draft(&lm_hidden, 42, 3)?;
// hidden_proj.forward() expects (1, 512)
// Gets (1, 256) instead
// Candle error or produces garbage embeddings
// Draft tokens are meaningless
```

**Attack Impact:**

- Inference error (garbage output)
- Model produces random/incorrect draft tokens
- Silent failure (no clear error message)

**Fix:**

```rust
pub fn draft(&self, lm_hidden: &Tensor, current_token: u32, num_steps: usize) -> Result<Vec<u32>> {
    let h0_squeezed = if lm_hidden.dims().len() == 3 {
        lm_hidden.squeeze(1)?
    } else {
        lm_hidden.clone()
    };

    // Validate shape before use
    let dims = h0_squeezed.dims();
    if dims.len() != 2 {
        return Err(Error::Msg(format!(
            "Expected 2D tensor (batch, d_model), got shape: {:?}",
            dims
        )));
    }
    if dims[1] != self.cfg.d_model {
        return Err(Error::Msg(format!(
            "Expected d_model={}, got {}",
            self.cfg.d_model, dims[1]
        )));
    }

    let h0 = self.hidden_proj.forward(&h0_squeezed)?;
    // ... rest of function
}
```

**Severity:** **P1** (Inference error, garbage output)

---

### P1-3: Command Injection Risk in GCE Watchdog (if VM Name Untrusted)

**File:** `train/gce/watchdog.sh`, Line 28
**Function:** VM monitoring loop

```bash
PREEMPTED=$(gcloud compute operations list \
    --project="$PROJECT" \
    --filter="operationType=compute.instances.preempted AND targetLink~${VM}" \
    --sort-by=~insertTime --limit=1 \
    --format="value(insertTime)" 2>/dev/null) || true
```

**Vulnerability:**

- `${VM}` is unquoted in gcloud filter string
- Currently hardcoded in `WATCH_VMS`, so **low risk**
- But if VM sourced from environment or config file, **command injection possible**

**Proof of Concept (if VM untrusted):**

```bash
VM="test' && echo 'PWNED' && echo '"
gcloud compute operations list \
    --filter="operationType=compute.instances.preempted AND targetLink~${VM}"
# Filter string becomes:
# "...targetLink~test' && echo 'PWNED' && echo '"
# Bash interprets && as command separator
```

**Current Risk:** **LOW** (VM hardcoded in script)

**If VM becomes user-input:** **P1** (Command injection)

**Fix:**

```bash
# Option 1: Validate VM name strictly
if [[ ! "$VM" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Invalid VM name: $VM" >&2
    continue
fi

# Option 2: Use gcloud API array syntax (safer)
gcloud compute operations list \
    --project="$PROJECT" \
    --filter="operationType=compute.instances.preempted AND targetLink:'instances/${VM}'" \
    ...

# Option 3: Quote the variable (partial mitigation)
--filter="operationType=compute.instances.preempted AND targetLink~'${VM}'"
```

**Severity:** **P1** (if VM is untrusted) / **P3** (currently hardcoded)

---

## MEDIUM-PRIORITY FINDINGS (P2 — Theoretical/Hardening)

### P2-1: Strict Aliasing Violation in Config Deserialization

**File:** `src/codec_12hz.c`, Line 397

```c
cfg.dec_ff_mult = *(float *)(cfg_buf + 44);
```

**Issue:** Cast from `uint8_t*` to `float*` violates C strict aliasing rules.
**Impact:** Undefined behavior, compiler optimizations may produce incorrect code.
**Fix:** Use `memcpy()`:

```c
float dec_ff_mult;
memcpy(&dec_ff_mult, cfg_buf + 44, sizeof(float));
cfg.dec_ff_mult = dec_ff_mult;
```

---

### P2-2: Silent Failure on Invalid dec_dim

**File:** `src/codec_12hz.c`, Line 323

```c
codec->output_weight = (float *)calloc((size_t)1 * (cfg->dec_dim / 16) * 7, sizeof(float));
```

**Issue:** No validation that `cfg->dec_dim >= 16`.
If `dec_dim = 8`, then `dec_dim / 16 = 0`, allocates 0 bytes.
**Impact:** Decoding silently produces garbage without error.
**Fix:**

```c
if (cfg->dec_dim < 16 || cfg->dec_dim % 16 != 0) {
    fprintf(stderr, "[codec_12hz] dec_dim must be >= 16 and divisible by 16\n");
    return NULL;
}
```

---

## PRIORITY SUMMARY

| Priority | Count | Item                                                                   |
| -------- | ----- | ---------------------------------------------------------------------- |
| **P0**   | 5     | Integer overflow (2), buffer overflow (1), path traversal (1), OOM (1) |
| **P1**   | 3     | Ring buffer logic (1), tensor validation (1), command injection (1)    |
| **P2**   | 2     | Strict aliasing (1), invalid config (1)                                |

## RECOMMENDATIONS

1. **Fix all P0 items immediately** before any deployment
2. **Fix all P1 items** before shipping to production
3. **Address P2 items** during next maintenance cycle
4. **Add comprehensive fuzzing** with malformed config structs
5. **Add negative test cases** for all public C functions (NULL pointers, extreme values)
