# Sonata TTS Evaluation Pipeline

## Quick Start

```bash
# Full evaluation (C pipeline + Python metrics)
make eval

# With UTMOS neural MOS (slower, downloads model)
make eval-full

# Just generate WAVs via C pipeline
make eval-generate

# Just run Python metrics on existing WAVs
make eval-python
```

## Reports

| File                         | Contents                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `reports/c_eval_report.json` | C pipeline: RTF, TTFA, WER (Conformer), prosody MOS, speaker consistency     |
| `reports/eval_report.json`   | Full: above + PESQ, STOI, MCD, F0, speaker sim, Whisper WER, SOTA comparison |

## Metrics

| Metric              | Source | Type                  | Good   | SOTA  |
| ------------------- | ------ | --------------------- | ------ | ----- |
| RTF                 | C      | Performance           | <0.2   | 0.03  |
| TTFA                | C      | Latency               | <300ms | 102ms |
| LM tok/s            | C      | Performance           | >40    | -     |
| Prosody MOS         | C      | Quality (ref-free)    | >3.5   | >4.0  |
| WER (Conformer)     | C      | Intelligibility       | <10%   | <3%   |
| Speaker consistency | C      | Voice identity        | >0.85  | -     |
| WER (Whisper)       | Python | Intelligibility       | <5%    | <0.9% |
| PESQ                | Python | Quality (ref)         | >3.0   | >3.5  |
| STOI                | Python | Intelligibility (ref) | >0.75  | >0.90 |
| MCD                 | Python | Spectral (ref)        | <6dB   | <4dB  |
| F0 corr             | Python | Prosody (ref)         | >0.70  | >0.85 |
| Speaker sim         | Python | Voice (ref)           | >0.75  | >0.90 |
| UTMOS               | Python | Quality (ref-free)    | >3.5   | >4.0  |

## Adding Reference Audio

Place reference WAVs in `eval/reference/` matching the naming `eval_NNNN.wav`.
This enables paired metrics (PESQ, STOI, MCD, F0, speaker sim).
