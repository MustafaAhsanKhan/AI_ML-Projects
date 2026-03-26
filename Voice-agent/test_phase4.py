"""
Phase 4 tests — Whisper STT

This file contains:
1) Deterministic async queue test (no model required)
2) Optional live microphone transcription (requires Whisper model files)

Usage:
  ./.venv/bin/python3 test_phase4.py
  ./.venv/bin/python3 test_phase4.py --live
"""

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd

sys.path.insert(0, "/Users/mustafa/Desktop/AI_ML-Projects/Voice-agent")

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

import config
from core.events import SpeechEndEvent, SpeechStartEvent
from stt.whisper_engine import WhisperEngine


async def test_whisper_run_loop_without_model() -> None:
    """
    Validates run-loop behavior deterministically without loading Whisper.
    """
    engine = WhisperEngine(model_size=config.WHISPER_MODEL_SIZE, model_path=config.WHISPER_MODEL_PATH)

    # Avoid loading the real model for deterministic queue behavior testing.
    engine._model = object()  # noqa: SLF001

    def fake_transcribe(audio: np.ndarray) -> str:
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        return "hello from stt"

    engine.transcribe = fake_transcribe  # type: ignore[method-assign]

    vad_event_q: asyncio.Queue = asyncio.Queue()
    stt_result_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(max_workers=1)
    task = asyncio.create_task(engine.run(vad_event_q, stt_result_q, executor, loop))

    try:
        # Non-SpeechEnd event should be ignored.
        await vad_event_q.put(
            SpeechStartEvent(pre_roll_audio=np.zeros(config.BLOCK_SIZE, dtype=np.float32))
        )

        # SpeechEnd event should trigger one transcription result.
        await vad_event_q.put(SpeechEndEvent(audio=np.ones(config.SAMPLE_RATE, dtype=np.float32)))

        await asyncio.wait_for(vad_event_q.join(), timeout=2.0)

        result = await asyncio.wait_for(stt_result_q.get(), timeout=2.0)
        assert result.text == "hello from stt"

        print("Whisper run-loop deterministic test OK")
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        executor.shutdown(wait=False)


def live_record(seconds: int = 5) -> np.ndarray:
    """Record one utterance from microphone for live STT smoke test."""
    print(f"Recording for {seconds}s... Speak now.")
    audio = sd.rec(
        int(seconds * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.reshape(-1).astype(np.float32)


def live_stt_smoke_test() -> None:
    """Optional end-to-end STT smoke test with microphone input."""
    engine = WhisperEngine(model_size=config.WHISPER_MODEL_SIZE, model_path=config.WHISPER_MODEL_PATH)
    engine.load()

    audio = live_record(seconds=5)
    text = engine.transcribe(audio)

    print("Transcription:")
    print(text if text else "<empty>")


if __name__ == "__main__":
    print("=== Phase 4: Whisper STT Tests ===\n")

    asyncio.run(test_whisper_run_loop_without_model())

    if "--live" in sys.argv:
        print("\n=== Live STT Smoke Test ===\n")
        live_stt_smoke_test()
    else:
        print("Skipping live test (re-run with --live when model is ready)")
