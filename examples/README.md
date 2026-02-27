# Examples

Usage examples for the Sonata HTTP API and WebSocket interface.

## Prerequisites

Start the Sonata server:

```bash
make
./sonata --server --server-port 8080
```

## Files

### `curl_tts.sh`

Shell script demonstrating the REST API with curl:

- Health check
- Basic TTS (plain text to WAV)
- OpenAI-compatible TTS request (JSON to Opus)
- Emotion-conditioned TTS
- Word timestamp extraction
- Custom output format (sample rate, encoding)
- Speech-to-text transcription

```bash
chmod +x curl_tts.sh
./curl_tts.sh
```

### `websocket_client.py`

Python WebSocket client for real-time voice interaction:

- Connects to `/v1/stream` via WebSocket
- Streams microphone audio to the server
- Receives and displays transcriptions, LLM tokens, and TTS audio

```bash
pip install websockets pyaudio
python websocket_client.py --host localhost --port 8080
```

## API Quick Reference

| Endpoint                   | Method | Description           |
| -------------------------- | ------ | --------------------- |
| `/health`                  | GET    | Health check          |
| `/v1/audio/speech`         | POST   | Text-to-speech        |
| `/v1/audio/transcriptions` | POST   | Speech-to-text        |
| `/v1/chat`                 | POST   | Chat with LLM         |
| `/v1/voices`               | POST   | Register cloned voice |
| `/v1/voices`               | GET    | List cloned voices    |
| `/v1/stream`               | GET    | WebSocket streaming   |

For full API documentation, see [docs/api.md](../docs/api.md).
