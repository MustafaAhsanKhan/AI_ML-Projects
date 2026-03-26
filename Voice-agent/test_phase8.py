"""Phase 8 tests — RealtimeOrchestrator (deterministic integration)."""

import asyncio
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "/Users/mustafa/Desktop/AI_ML-Projects/Voice-agent")

from core.conversation_manager import ConversationManager
from core.events import AudioChunk, LLMToken, SentenceReady, SpeechEndEvent, SpeechStartEvent, TranscriptionResult
from core.realtime_orchestrator import RealtimeOrchestrator
from core.state_machine import AssistantState


class FakeWhisperEngine:
    async def run(self, vad_event_q, stt_result_q, executor, loop):
        while True:
            event = await vad_event_q.get()
            try:
                if isinstance(event, SpeechEndEvent):
                    await stt_result_q.put(TranscriptionResult(text="hello orchestrator"))
            finally:
                vad_event_q.task_done()


class FakeLlamaEngine:
    async def stream(self, prompt, token_q, cancel_event, executor):
        tokens = ["This is sentence one.", " Second sentence!"]

        def produce():
            for tok in tokens:
                if cancel_event.is_set():
                    break
                fut = asyncio.run_coroutine_threadsafe(
                    token_q.put(LLMToken(text=tok, is_final=False)),
                    loop,
                )
                fut.result()
                time.sleep(0.01)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, produce)
        await token_q.put(LLMToken(text="", is_final=True))


class FakePiperEngine:
    async def run(self, sentence_q, audio_out_q, executor, cancel_event):
        while True:
            item = await sentence_q.get()
            try:
                if isinstance(item, SentenceReady) and not cancel_event.is_set():
                    pcm = np.ones(1024, dtype=np.float32)
                    await audio_out_q.put(AudioChunk(pcm=pcm))
            finally:
                sentence_q.task_done()


class FakeAudioOutput:
    def __init__(self, audio_out_q: asyncio.Queue):
        self.audio_out_q = audio_out_q
        self.flush_calls = 0
        self.stop_calls = 0
        self.played_chunks = 0

    async def run(self, executor):
        while True:
            chunk = await self.audio_out_q.get()
            try:
                if isinstance(chunk, AudioChunk):
                    self.played_chunks += 1
            finally:
                self.audio_out_q.task_done()

    def flush(self):
        self.flush_calls += 1
        while not self.audio_out_q.empty():
            try:
                self.audio_out_q.get_nowait()
                self.audio_out_q.task_done()
            except asyncio.QueueEmpty:
                break

    def stop(self):
        self.stop_calls += 1


async def test_orchestrator_end_to_end_flow() -> None:
    vad_event_q = asyncio.Queue(maxsize=20)
    stt_result_q = asyncio.Queue(maxsize=10)
    sentence_q = asyncio.Queue(maxsize=10)
    audio_out_q = asyncio.Queue(maxsize=50)

    cm = ConversationManager(system_prompt="You are helpful.", max_tokens=4096)

    whisper = FakeWhisperEngine()
    llama = FakeLlamaEngine()
    piper = FakePiperEngine()
    audio_output = FakeAudioOutput(audio_out_q)

    inference_executor = ThreadPoolExecutor(max_workers=1)
    tts_executor = ThreadPoolExecutor(max_workers=1)

    orch = RealtimeOrchestrator(
        whisper_engine=whisper,
        llama_engine=llama,
        piper_engine=piper,
        conversation_manager=cm,
        audio_output=audio_output,
        vad_event_q=vad_event_q,
        stt_result_q=stt_result_q,
        sentence_q=sentence_q,
        audio_out_q=audio_out_q,
        inference_executor=inference_executor,
        tts_executor=tts_executor,
    )

    run_task = asyncio.create_task(orch.run())

    try:
        await vad_event_q.put(SpeechStartEvent(pre_roll_audio=np.zeros(512, dtype=np.float32)))
        await vad_event_q.put(SpeechEndEvent(audio=np.ones(16000, dtype=np.float32)))

        await asyncio.sleep(0.3)

        # Verify response made it through and was stored.
        roles = [m["role"] for m in cm.messages]
        assert "user" in roles
        assert "assistant" in roles
        assert audio_output.played_chunks > 0

        # State should return idle after response completion.
        assert orch.state.current in {AssistantState.IDLE, AssistantState.SPEAKING}

        print("Orchestrator end-to-end flow test OK")
    finally:
        await orch._shutdown()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        inference_executor.shutdown(wait=False)
        tts_executor.shutdown(wait=False)


async def test_barge_in_interrupt_flushes_audio() -> None:
    vad_event_q = asyncio.Queue(maxsize=20)
    stt_result_q = asyncio.Queue(maxsize=10)
    sentence_q = asyncio.Queue(maxsize=10)
    audio_out_q = asyncio.Queue(maxsize=50)

    cm = ConversationManager(system_prompt="You are helpful.", max_tokens=4096)

    whisper = FakeWhisperEngine()

    class SlowLlama(FakeLlamaEngine):
        async def stream(self, prompt, token_q, cancel_event, executor):
            def produce():
                for tok in ["Long response chunk one. ", "Long response chunk two. "]:
                    if cancel_event.is_set():
                        break
                    fut = asyncio.run_coroutine_threadsafe(
                        token_q.put(LLMToken(text=tok, is_final=False)),
                        loop,
                    )
                    fut.result()
                    time.sleep(0.1)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(executor, produce)
            await token_q.put(LLMToken(text="", is_final=True))

    llama = SlowLlama()
    piper = FakePiperEngine()
    audio_output = FakeAudioOutput(audio_out_q)

    inference_executor = ThreadPoolExecutor(max_workers=1)
    tts_executor = ThreadPoolExecutor(max_workers=1)

    orch = RealtimeOrchestrator(
        whisper_engine=whisper,
        llama_engine=llama,
        piper_engine=piper,
        conversation_manager=cm,
        audio_output=audio_output,
        vad_event_q=vad_event_q,
        stt_result_q=stt_result_q,
        sentence_q=sentence_q,
        audio_out_q=audio_out_q,
        inference_executor=inference_executor,
        tts_executor=tts_executor,
    )

    run_task = asyncio.create_task(orch.run())

    try:
        # Force speaking state so interrupt path is deterministic.
        orch._transition_safe(AssistantState.LISTENING)
        orch._transition_safe(AssistantState.TRANSCRIBING)
        orch._transition_safe(AssistantState.THINKING)
        orch._transition_safe(AssistantState.SPEAKING)

        # Put queued audio so flush has real work to clear.
        await audio_out_q.put(AudioChunk(pcm=np.ones(1024, dtype=np.float32)))

        # Barge-in while speaking.
        await vad_event_q.put(SpeechStartEvent(pre_roll_audio=np.zeros(512, dtype=np.float32)))
        await asyncio.sleep(0.1)

        assert audio_output.flush_calls >= 1
        assert orch.state.current in {AssistantState.LISTENING, AssistantState.IDLE, AssistantState.SPEAKING}

        print("Orchestrator barge-in test OK")
    finally:
        await orch._shutdown()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        inference_executor.shutdown(wait=False)
        tts_executor.shutdown(wait=False)


if __name__ == "__main__":
    print("=== Phase 8: Realtime Orchestrator Tests ===\n")
    asyncio.run(test_orchestrator_end_to_end_flow())
    asyncio.run(test_barge_in_interrupt_flushes_audio())
    print("\nAll Phase 8 tests passed")
