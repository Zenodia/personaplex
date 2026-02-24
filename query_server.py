#!/usr/bin/env python3
"""
PersonaPlex server query client: send a local WAV file to the server and save the
response audio to a WAV file (and optionally transcript to a text file).

Usage:
  python query_server.py --input-wav path/to/input.wav --output-wav path/to/response.wav
  python query_server.py -i input.wav -o response.wav --voice-prompt NATF2.pt --text-prompt "You enjoy having a good conversation."

Requires the PersonaPlex server to be running (e.g. via Docker or `python -m moshi.server --ssl ...`).
Uses the same protocol as the Web UI: WebSocket to /api/chat with Opus-encoded audio.
"""

from __future__ import annotations

import argparse
import asyncio
import ssl
import sys
from pathlib import Path
from urllib.parse import quote

import aiohttp
import numpy as np
import sphn


# Server expects 24 kHz mono; frame size matches server's processing (24000 / 12.5)
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples


def load_and_prepare_wav(path: str) -> np.ndarray:
    """Load WAV, resample to SAMPLE_RATE, convert to mono. Returns 1D float32 array."""
    pcm, sr = sphn.read(path)
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=0)
    elif pcm.ndim == 1:
        pass
    else:
        pcm = pcm.reshape(-1).mean(axis=0)
    if sr != SAMPLE_RATE:
        # sphn.resample expects (C, T)
        pcm = pcm[np.newaxis, :]
        pcm = sphn.resample(pcm, src_sample_rate=sr, dst_sample_rate=SAMPLE_RATE)
        pcm = pcm[0]
    return pcm.astype(np.float32)


def encode_pcm_to_opus_chunks(pcm: np.ndarray) -> list[bytes]:
    """Encode PCM (1D float32 at SAMPLE_RATE) to Opus chunks matching server frame size."""
    writer = sphn.OpusStreamWriter(SAMPLE_RATE)
    chunks = []
    n = len(pcm)
    for start in range(0, n, FRAME_SIZE):
        end = min(start + FRAME_SIZE, n)
        frame = pcm[start:end]
        if len(frame) < FRAME_SIZE:
            frame = np.pad(frame, (0, FRAME_SIZE - len(frame)), mode="constant", constant_values=0)
        writer.append_pcm(frame)
        while True:
            data = writer.read_bytes()
            if not data:
                break
            chunks.append(data)
    return chunks


def decode_opus_to_pcm(opus_payloads: list[bytes]) -> np.ndarray:
    """Decode a list of Opus payloads (from server) into one PCM float32 array."""
    reader = sphn.OpusStreamReader(SAMPLE_RATE)
    parts = []
    for payload in opus_payloads:
        reader.append_bytes(payload)
        while True:
            pcm = reader.read_pcm()
            if pcm.size == 0:
                break
            if pcm.ndim > 1:
                pcm = pcm.ravel()
            parts.append(pcm)
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


async def run(
    input_wav: str,
    output_wav: str,
    output_text: str | None,
    server_url: str,
    voice_prompt: str,
    text_prompt: str,
    seed: int | None,
    insecure: bool,
    send_delay: float,
    receive_timeout: float,
) -> None:
    pcm = load_and_prepare_wav(input_wav)
    opus_chunks = encode_pcm_to_opus_chunks(pcm)
    duration_sec = len(pcm) / SAMPLE_RATE
    print(f"Loaded {input_wav}: {duration_sec:.2f}s, {len(opus_chunks)} Opus chunks to send")

    params = [
        ("voice_prompt", voice_prompt),
        ("text_prompt", text_prompt),
    ]
    if seed is not None:
        params.append(("seed", str(seed)))
    query = "&".join(f"{k}={quote(str(v))}" for k, v in params)
    ws_url = f"{server_url}?{query}"

    ssl_context = None
    if ws_url.startswith("wss://") and insecure:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    audio_payloads: list[bytes] = []
    transcript_parts: list[str] = []

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            ws_url,
            ssl=ssl_context,
            heartbeat=30.0,
        ) as ws:
            # Wait for handshake (single byte 0x00)
            msg = await ws.receive()
            if msg.type != aiohttp.WSMsgType.BINARY or msg.data != b"\x00":
                print("Unexpected handshake:", msg, file=sys.stderr)
                return
            print("Connected; sending input audio...")

            async def send_audio():
                for i, chunk in enumerate(opus_chunks):
                    await ws.send_bytes(b"\x01" + chunk)
                    if send_delay > 0:
                        await asyncio.sleep(send_delay)
                print("Finished sending input audio.")

            async def receive_loop():
                nonlocal audio_payloads, transcript_parts
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.CLOSED:
                            break
                        if msg.type == aiohttp.WSMsgType.ERROR:
                            print("WebSocket error:", ws.exception(), file=sys.stderr)
                            break
                        if msg.type != aiohttp.WSMsgType.BINARY or not msg.data:
                            continue
                        kind = msg.data[0]
                        payload = bytes(msg.data[1:])
                        if kind == 0x01:
                            audio_payloads.append(payload)
                        elif kind == 0x02:
                            transcript_parts.append(payload.decode("utf-8", errors="replace"))
                except asyncio.CancelledError:
                    pass

            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(receive_loop())
            await send_task
            # Allow time for server to stream back response
            try:
                await asyncio.wait_for(asyncio.shield(recv_task), timeout=receive_timeout)
            except asyncio.TimeoutError:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
                print("Receive timeout reached; closing connection.")

    response_pcm = decode_opus_to_pcm(audio_payloads)
    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(output_wav, response_pcm, SAMPLE_RATE)
    print(f"Wrote response audio to {output_wav} ({len(response_pcm) / SAMPLE_RATE:.2f}s)")

    if output_text and transcript_parts:
        text_path = Path(output_text)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("".join(transcript_parts), encoding="utf-8")
        print(f"Wrote transcript to {output_text}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Send a WAV file to the PersonaPlex server and save the response as WAV (and optional transcript).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input-wav", "-i",
        required=True,
        help="Path to input WAV file (user speech).",
    )
    ap.add_argument(
        "--output-wav", "-o",
        required=True,
        help="Path to write response WAV file.",
    )
    ap.add_argument(
        "--output-text", "-t",
        default=None,
        help="Path to write response transcript (optional).",
    )
    ap.add_argument(
        "--server",
        default="wss://localhost:8998",
        help="Base URL of the PersonaPlex server (e.g. wss://localhost:8998).",
    )
    ap.add_argument(
        "--voice-prompt", "-v",
        default="NATF2.pt",
        help="Voice prompt filename (e.g. NATF2.pt, NATM1.pt).",
    )
    ap.add_argument(
        "--text-prompt", "-p",
        default="You enjoy having a good conversation.",
        help="System text prompt for the assistant.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional).",
    )
    ap.add_argument(
        "--insecure", "-k",
        action="store_true",
        help="Do not verify SSL certificate (use for self-signed / local server).",
    )
    ap.add_argument(
        "--send-delay",
        type=float,
        default=0.02,
        help="Delay in seconds between sending Opus chunks (simulate streaming).",
    )
    ap.add_argument(
        "--receive-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for response after sending input.",
    )
    args = ap.parse_args()

    base = args.server.rstrip("/")
    url = f"{base}/api/chat"

    asyncio.run(
        run(
            input_wav=args.input_wav,
            output_wav=args.output_wav,
            output_text=args.output_text,
            server_url=url,
            voice_prompt=args.voice_prompt,
            text_prompt=args.text_prompt,
            seed=args.seed,
            insecure=args.insecure,
            send_delay=args.send_delay,
            receive_timeout=args.receive_timeout,
        )
    )


if __name__ == "__main__":
    main()
