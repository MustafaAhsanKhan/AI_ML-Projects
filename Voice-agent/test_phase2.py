"""
Phase 2 tests — Audio Layer

Tests are split into two sections:
  1. Unit tests for AudioBuffer (no hardware required)
  2. Passthrough smoke test: mic → speaker (requires real audio hardware)
     Run with:  python3 test_phase2.py --passthrough
"""
import sys
import asyncio
import logging

import numpy as np

sys.path.insert(0, "/Users/mustafa/Desktop/AI_ML-Projects/Voice-agent")
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

import config
from audio.audio_buffer import AudioBuffer
from core.events import AudioChunk


# ---------------------------------------------------------------------------
# AudioBuffer unit tests
# ---------------------------------------------------------------------------

def test_audio_buffer_happy_path():
    buf = AudioBuffer(pre_roll_chunks=3)
    chunk = np.ones(config.BLOCK_SIZE, dtype=np.float32)

    # Feed 5 chunks before recording starts (only last 3 kept as pre-roll)
    for i in range(5):
        buf.add_chunk(chunk * i)

    assert not buf.is_recording

    buf.start_recording()
    assert buf.is_recording

    # Pre-roll should have seeded 3 chunks (values 2, 3, 4)
    # Feed 2 more chunks after recording starts
    buf.add_chunk(chunk * 10)
    buf.add_chunk(chunk * 11)

    audio = buf.stop_recording()

    assert not buf.is_recording
    # Total: 3 pre-roll + 2 recorded = 5 chunks
    assert len(audio) == 5 * config.BLOCK_SIZE
    assert audio.dtype == np.float32
    print("AudioBuffer happy path OK")


def test_audio_buffer_pre_roll_values():
    buf = AudioBuffer(pre_roll_chunks=2)
    chunk = np.ones(config.BLOCK_SIZE, dtype=np.float32)

    buf.add_chunk(chunk * 1)  # will be evicted
    buf.add_chunk(chunk * 2)  # pre-roll[0]
    buf.add_chunk(chunk * 3)  # pre-roll[1]

    buf.start_recording()
    buf.add_chunk(chunk * 4)  # recorded
    audio = buf.stop_recording()

    # Expect: [2, 2, 2...] [3, 3, 3...] [4, 4, 4...]
    assert len(audio) == 3 * config.BLOCK_SIZE
    assert np.allclose(audio[:config.BLOCK_SIZE], 2.0)
    assert np.allclose(audio[config.BLOCK_SIZE:2*config.BLOCK_SIZE], 3.0)
    assert np.allclose(audio[2*config.BLOCK_SIZE:], 4.0)
    print("AudioBuffer pre-roll value ordering OK")


def test_audio_buffer_reset():
    buf = AudioBuffer(pre_roll_chunks=2)
    chunk = np.ones(config.BLOCK_SIZE, dtype=np.float32)

    for _ in range(4):
        buf.add_chunk(chunk)

    buf.start_recording()
    buf.add_chunk(chunk)
    assert buf.is_recording
    assert buf.buffered_seconds > 0

    buf.reset()
    assert not buf.is_recording
    assert buf.buffered_seconds == 0.0
    print("AudioBuffer reset OK")


def test_audio_buffer_stop_without_start():
    buf = AudioBuffer()
    result = buf.stop_recording()
    assert len(result) == 0
    print("AudioBuffer stop-without-start returns empty array OK")


def test_audio_buffer_empty_recording():
    buf = AudioBuffer(pre_roll_chunks=0)
    buf.start_recording()
    audio = buf.stop_recording()
    assert len(audio) == 0
    print("AudioBuffer empty recording OK")


# ---------------------------------------------------------------------------
# Passthrough smoke test (requires --passthrough flag + audio hardware)
# ---------------------------------------------------------------------------

async def passthrough_test():
    """
    Captures 3 seconds from the microphone and plays it back immediately.
    Verifies that AudioInputStream and AudioOutputStream work end-to-end.
    """
    from audio.input_stream import AudioInputStream
    from audio.output_stream import AudioOutputStream
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    audio_in_q: asyncio.Queue = asyncio.Queue(maxsize=200)
    audio_out_q: asyncio.Queue = asyncio.Queue(maxsize=200)

    input_stream = AudioInputStream(queue=audio_in_q, loop=loop)
    output_stream = AudioOutputStream(queue=audio_out_q, loop=loop)

    executor = ThreadPoolExecutor(max_workers=1)

    input_stream.start()
    output_stream.start()

    print("Passthrough: capturing 3 seconds — speak into the mic...")

    output_task = asyncio.create_task(output_stream.run(executor))

    # Capture 3 seconds worth of chunks
    total_chunks = int(3 * config.SAMPLE_RATE / config.BLOCK_SIZE)
    for _ in range(total_chunks):
        chunk = await audio_in_q.get()
        # Resample from 16kHz (input) to 22050Hz (output) via repeat interleave
        # For a quick smoke test we just play raw at the output rate (will be pitch-shifted)
        # A proper resample is done in the TTS pipeline
        await audio_out_q.put(AudioChunk(pcm=chunk))

    # Drain remaining output
    await audio_out_q.join()

    output_task.cancel()
    input_stream.stop()
    output_stream.stop()
    executor.shutdown(wait=False)
    print("Passthrough smoke test complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Phase 2: Unit Tests ===\n")
    test_audio_buffer_happy_path()
    test_audio_buffer_pre_roll_values()
    test_audio_buffer_reset()
    test_audio_buffer_stop_without_start()
    test_audio_buffer_empty_recording()

    print("\n=== All AudioBuffer unit tests passed ===\n")

    if "--passthrough" in sys.argv:
        print("=== Phase 2: Passthrough Test ===\n")
        asyncio.run(passthrough_test())
    else:
        print("Skipping passthrough test (re-run with --passthrough to test audio hardware)")
