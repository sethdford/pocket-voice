#!/usr/bin/env bash
# Sonata TTS API examples using curl
#
# Prerequisites:
#   1. Build Sonata: make
#   2. Start the server: ./sonata --server --server-port 8080
#
# Optional: Set SONATA_API_KEY for authenticated access.

BASE_URL="${SONATA_URL:-http://localhost:8080}"

auth_args=()
if [ -n "$SONATA_API_KEY" ]; then
    auth_args=(-H "Authorization: Bearer $SONATA_API_KEY")
fi

echo "=== Health Check ==="
curl -s "${auth_args[@]}" "$BASE_URL/health" | python3 -m json.tool
echo

echo "=== Basic TTS (plain text → WAV) ==="
curl -s -X POST "${auth_args[@]}" "$BASE_URL/v1/audio/speech" \
  -d "Hello, this is Sonata speaking." \
  -o basic_output.wav
echo "Saved: basic_output.wav"
echo

echo "=== OpenAI-Compatible TTS (JSON → Opus) ==="
curl -s -X POST "${auth_args[@]}" "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Sonata runs entirely on Apple Silicon.",
    "voice": "alloy",
    "response_format": "opus"
  }' \
  -o openai_compat.opus
echo "Saved: openai_compat.opus"
echo

echo "=== TTS with Emotion ==="
curl -s -X POST "${auth_args[@]}" "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am so excited to show you what Sonata can do!",
    "emotion": "excited",
    "speed": 1.1
  }' \
  -o excited_output.wav
echo "Saved: excited_output.wav"
echo

echo "=== TTS with Word Timestamps ==="
curl -s -X POST "${auth_args[@]}" "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Each word gets a timestamp.",
    "word_timestamps": true
  }' | python3 -m json.tool
echo

echo "=== TTS with Custom Output Format ==="
curl -s -X POST "${auth_args[@]}" "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "High quality 48kHz output.",
    "output_format": {
      "sample_rate": 48000,
      "encoding": "pcm_f32le",
      "container": "wav"
    }
  }' \
  -o hq_output.wav
echo "Saved: hq_output.wav"
echo

echo "=== STT (WAV → Text) ==="
if [ -f "basic_output.wav" ]; then
    curl -s -X POST "${auth_args[@]}" "$BASE_URL/v1/audio/transcriptions" \
      -H "Content-Type: audio/wav" \
      --data-binary @basic_output.wav | python3 -m json.tool
else
    echo "Skipped: no WAV file to transcribe (run TTS examples first)"
fi
echo

echo "Done. Clean up with: rm -f basic_output.wav openai_compat.opus excited_output.wav hq_output.wav"
