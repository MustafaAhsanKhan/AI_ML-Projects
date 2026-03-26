"""
Phase 5 tests — Llama streaming

This file contains deterministic tests that do not require llama-cpp model loading.

Usage:
  ./.venv/bin/python3 test_phase5.py
"""

import asyncio
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/Users/mustafa/Desktop/AI_ML-Projects/Voice-agent")

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

from core.events import LLMToken
from llm.llama_engine import LlamaEngine


async def collect_tokens(token_q: asyncio.Queue) -> list[LLMToken]:
    collected: list[LLMToken] = []
    while True:
        tok: LLMToken = await token_q.get()
        collected.append(tok)
        token_q.task_done()
        if tok.is_final:
            return collected


async def test_stream_happy_path() -> None:
    engine = LlamaEngine(model_path=Path("/tmp/not-used.gguf"))
    engine._model = object()  # noqa: SLF001

    def fake_stream_sync(_prompt: str, _cancel_event: threading.Event):
        yield "Hello"
        yield " "
        yield "world"

    engine._stream_sync = fake_stream_sync  # type: ignore[method-assign]

    token_q: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        producer_task = asyncio.create_task(
            engine.stream("test", token_q, cancel_event, executor)
        )

        tokens = await collect_tokens(token_q)
        await producer_task

        text = "".join(t.text for t in tokens if not t.is_final)
        assert text == "Hello world"
        assert tokens[-1].is_final is True
        print("LLM stream happy-path test OK")
    finally:
        executor.shutdown(wait=False)


async def test_stream_cancellation() -> None:
    engine = LlamaEngine(model_path=Path("/tmp/not-used.gguf"))
    engine._model = object()  # noqa: SLF001

    def fake_stream_sync(_prompt: str, cancel_event: threading.Event):
        for tok in ["one", " ", "two", " ", "three"]:
            if cancel_event.is_set():
                return
            time.sleep(0.02)
            yield tok

    engine._stream_sync = fake_stream_sync  # type: ignore[method-assign]

    token_q: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        producer_task = asyncio.create_task(
            engine.stream("test", token_q, cancel_event, executor)
        )

        await asyncio.sleep(0.05)
        cancel_event.set()

        tokens = await collect_tokens(token_q)
        await producer_task

        assert tokens[-1].is_final is True
        generated_text = "".join(t.text for t in tokens if not t.is_final)
        assert len(generated_text) > 0
        print("LLM stream cancellation test OK")
    finally:
        executor.shutdown(wait=False)


if __name__ == "__main__":
    print("=== Phase 5: LLM Streaming Tests ===\n")
    asyncio.run(test_stream_happy_path())
    asyncio.run(test_stream_cancellation())
    print("\nAll Phase 5 deterministic tests passed")
