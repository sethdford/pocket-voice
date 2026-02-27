#!/usr/bin/env python3
"""
Sonata WebSocket streaming client.

Connects to the Sonata server via WebSocket for real-time voice interaction.
Sends audio from your microphone and receives transcriptions, LLM responses,
and TTS audio.

Prerequisites:
    pip install websockets pyaudio

Usage:
    # Start the Sonata server first:
    ./sonata --server --server-port 8080

    # Then run this client:
    python websocket_client.py
    python websocket_client.py --host localhost --port 8080
"""

import argparse
import asyncio
import json
import struct
import sys

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)

try:
    import pyaudio
except ImportError:
    pyaudio = None


SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 4800  # 200ms at 24kHz
FORMAT_F32 = 1  # pyaudio.paFloat32


async def stream_microphone(ws):
    """Capture audio from microphone and send via WebSocket."""
    if pyaudio is None:
        print("[mic] pyaudio not installed — skipping mic capture")
        print("[mic] Install with: pip install pyaudio")
        return

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    print("[mic] Recording... (speak into your microphone)")
    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            await ws.send(data)
            await asyncio.sleep(0)  # yield to event loop
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("[mic] Stopped")


async def receive_events(ws):
    """Receive and handle server events."""
    try:
        async for message in ws:
            if isinstance(message, str):
                # JSON text event
                event = json.loads(message)
                event_type = event.get("type", "unknown")

                if event_type == "listening":
                    print("\n[server] Listening...")
                elif event_type == "transcript":
                    print(f"\n[stt] {event.get('text', '')}")
                elif event_type == "llm_token":
                    print(event.get("text", ""), end="", flush=True)
                elif event_type == "speaking":
                    print("\n[server] Speaking...")
                else:
                    print(f"\n[server] {event}")

            elif isinstance(message, bytes):
                # Binary audio data from TTS
                n_samples = len(message) // 4  # float32
                print(f"[audio] Received {n_samples} samples "
                      f"({n_samples / SAMPLE_RATE:.2f}s)")
                # To play this audio, you would write it to a PyAudio output stream

    except websockets.exceptions.ConnectionClosed:
        print("\n[server] Connection closed")


async def main(host: str, port: int):
    uri = f"ws://{host}:{port}/v1/stream"
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as ws:
            print("Connected!")

            # Run mic capture and event receiver concurrently
            mic_task = asyncio.create_task(stream_microphone(ws))
            recv_task = asyncio.create_task(receive_events(ws))

            try:
                await asyncio.gather(mic_task, recv_task)
            except KeyboardInterrupt:
                print("\nDisconnecting...")
                mic_task.cancel()
                recv_task.cancel()

    except ConnectionRefusedError:
        print(f"Could not connect to {uri}")
        print("Make sure the Sonata server is running:")
        print("  ./sonata --server --server-port 8080")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sonata WebSocket client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        print("\nBye!")
