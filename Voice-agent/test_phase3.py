"""
Phase 3 tests — Silero VAD

This file contains:
1) A deterministic async behavior test (no ONNX model required)
2) An optional live smoke test (requires Silero model + microphone)

Usage:
  ./.venv/bin/python3 test_phase3.py
  ./.venv/bin/python3 test_phase3.py --live
"""

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/mustafa/Desktop/AI_ML-Projects/Voice-agent")

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

import config
from audio.input_stream import AudioInputStream
from core.events import SpeechEndEvent, SpeechStartEvent
from vad.silero_vad import SileroVAD


def run_async(coro):
    return asyncio.run(coro)


async def test_vad_behavior_without_model() -> None:
    """
    Verifies event timing and payload contracts using deterministic probabilities.
    This does not execute ONNX inference.
    """

    vad = SileroVAD(model_path=Path("/tmp/not-used.onnx"))

    # Keep the test short by reducing silence window.
    vad._silence_chunks = 2  # noqa: SLF001
    vad._session = object()  # noqa: SLF001

    probs = iter([0.1, 0.2, 0.8, 0.9, 0.3, 0.2, 0.1])

    def fake_process_chunk(_chunk: np.ndarray) -> float:
        return next(probs)

    vad.process_chunk = fake_process_chunk  # type: ignore[method-assign]

    audio_in_q: asyncio.Queue = asyncio.Queue()
    vad_event_q: asyncio.Queue = asyncio.Queue()

    task = asyncio.create_task(vad.run(audio_in_q, vad_event_q))

    try:
        for _ in range(7):
            await audio_in_q.put(np.ones(config.BLOCK_SIZE, dtype=np.float32))

        await asyncio.wait_for(audio_in_q.join(), timeout=2.0)
        await asyncio.sleep(0.05)

        events = []
        while not vad_event_q.empty():
            events.append(vad_event_q.get_nowait())

        assert any(isinstance(e, SpeechStartEvent) for e in events), "Missing SpeechStartEvent"
        assert any(isinstance(e, SpeechEndEvent) for e in events), "Missing SpeechEndEvent"

        start_event = next(e for e in events if isinstance(e, SpeechStartEvent))
        end_event = next(e for e in events if isinstance(e, SpeechEndEvent))

        assert start_event.pre_roll_audio.dtype == np.float32
        assert end_event.audio.dtype == np.float32
        assert len(end_event.audio) > 0

        print("VAD deterministic behavior test OK")
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def live_vad_smoke_test() -> None:
    """
    Optional microphone test. Requires a valid Silero ONNX model file.
    Logs speech start/end events for 10 seconds.
    """

    if not config.SILERO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Silero model not found at {config.SILERO_MODEL_PATH}. "
            "Download the model before running --live."
        )

    loop = asyncio.get_event_loop()
    audio_in_q: asyncio.Queue = asyncio.Queue(maxsize=200)
    vad_event_q: asyncio.Queue = asyncio.Queue(maxsize=50)

    mic = AudioInputStream(queue=audio_in_q, loop=loop)
    vad = SileroVAD(model_path=config.SILERO_MODEL_PATH)
    vad.load()

    mic.start()
    vad_task = asyncio.create_task(vad.run(audio_in_q, vad_event_q))

    print("Live VAD smoke test started (10s). Speak naturally, then pause.")

    try:
        end_time = loop.time() + 10.0
        while loop.time() < end_time:
            timeout = max(0.0, end_time - loop.time())
            try:
                event = await asyncio.wait_for(vad_event_q.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            if isinstance(event, SpeechStartEvent):
                print(f"SpeechStartEvent: pre-roll samples={len(event.pre_roll_audio)}")
            elif isinstance(event, SpeechEndEvent):
                duration_s = len(event.audio) / config.SAMPLE_RATE
                print(
                    f"SpeechEndEvent: samples={len(event.audio)} "
                    f"duration={duration_s:.2f}s"
                )

    finally:
        vad_task.cancel()
        mic.stop()
        try:
            await vad_task
        except asyncio.CancelledError:
            pass

    print("Live VAD smoke test complete")


if __name__ == "__main__":
    print("=== Phase 3: Silero VAD Tests ===\n")

    run_async(test_vad_behavior_without_model())

    if "--live" in sys.argv:
        print("\n=== Live VAD Smoke Test ===\n")
        run_async(live_vad_smoke_test())
    else:
        print("Skipping live test (re-run with --live after model download)")
