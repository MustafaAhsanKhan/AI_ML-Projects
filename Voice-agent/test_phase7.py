"""Phase 7 tests — PiperEngine (deterministic)."""

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/mustafa/Desktop/AI_ML-Projects/Voice-agent")

import config
from core.events import SentenceReady
from tts.piper_engine import PiperEngine


def test_chunk_audio_padding_and_sizes() -> None:
    engine = PiperEngine(Path("/tmp/m.onnx"), Path("/tmp/m.onnx.json"))

    audio = np.linspace(-0.5, 0.5, config.OUTPUT_CHUNK_SIZE + 200, dtype=np.float32)
    chunks = engine.chunk_audio(audio)

    assert len(chunks) == 2
    assert len(chunks[0]) == config.OUTPUT_CHUNK_SIZE
    assert len(chunks[1]) == config.OUTPUT_CHUNK_SIZE
    assert np.allclose(chunks[1][:200], audio[-200:])
    assert np.allclose(chunks[1][200:], 0.0)
    print("TTS chunking/padding test OK")


def test_resample_if_needed_identity_and_change() -> None:
    engine = PiperEngine(Path("/tmp/m.onnx"), Path("/tmp/m.onnx.json"))
    audio = np.sin(np.linspace(0, 8, 22050)).astype(np.float32)

    same = engine._resample_if_needed(audio, config.TTS_SAMPLE_RATE)  # noqa: SLF001
    changed = engine._resample_if_needed(audio, 16000)  # noqa: SLF001

    assert len(same) == len(audio)
    assert len(changed) != 0
    assert abs(len(changed) - len(audio) * config.TTS_SAMPLE_RATE / 16000) < 10
    print("TTS resampling test OK")


async def test_tts_run_loop_with_cancellation() -> None:
    engine = PiperEngine(Path("/tmp/m.onnx"), Path("/tmp/m.onnx.json"))
    engine._voice = object()  # noqa: SLF001

    def fake_synthesize(text: str) -> np.ndarray:
        assert text
        return np.ones(config.OUTPUT_CHUNK_SIZE * 2, dtype=np.float32)

    engine.synthesize = fake_synthesize  # type: ignore[method-assign]

    sentence_q: asyncio.Queue = asyncio.Queue()
    audio_out_q: asyncio.Queue = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=1)
    cancel_event = threading.Event()

    task = asyncio.create_task(engine.run(sentence_q, audio_out_q, executor, cancel_event))

    try:
        await sentence_q.put(SentenceReady(text="hello"))
        await asyncio.wait_for(sentence_q.join(), timeout=2.0)
        produced_1 = audio_out_q.qsize()
        assert produced_1 >= 2

        cancel_event.set()
        await sentence_q.put(SentenceReady(text="skip me"))
        await asyncio.wait_for(sentence_q.join(), timeout=2.0)
        produced_2 = audio_out_q.qsize()
        assert produced_2 == produced_1

        print("TTS run-loop cancellation test OK")
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        executor.shutdown(wait=False)


if __name__ == "__main__":
    print("=== Phase 7: Piper TTS Tests ===\n")
    test_chunk_audio_padding_and_sizes()
    test_resample_if_needed_identity_and_change()
    asyncio.run(test_tts_run_loop_with_cancellation())
    print("\nAll Phase 7 tests passed")
