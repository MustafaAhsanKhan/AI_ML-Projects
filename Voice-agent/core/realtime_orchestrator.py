import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import config
from core.events import LLMToken, SpeechStartEvent, TranscriptionResult, SentenceReady
from core.state_machine import AssistantState, InvalidTransitionError, StateMachine

logger = logging.getLogger(__name__)


class RealtimeOrchestrator:
    """Central coordinator for STT -> LLM -> TTS with barge-in handling."""

    def __init__(
        self,
        *,
        whisper_engine,
        llama_engine,
        piper_engine,
        conversation_manager,
        audio_output,
        vad_event_q: asyncio.Queue,
        stt_result_q: asyncio.Queue,
        sentence_q: asyncio.Queue,
        audio_out_q: asyncio.Queue,
        inference_executor: ThreadPoolExecutor,
        tts_executor: ThreadPoolExecutor,
        output_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.whisper_engine = whisper_engine
        self.llama_engine = llama_engine
        self.piper_engine = piper_engine
        self.conversation_manager = conversation_manager
        self.audio_output = audio_output

        self.vad_event_q = vad_event_q
        self.stt_result_q = stt_result_q
        self.sentence_q = sentence_q
        self.audio_out_q = audio_out_q

        self.inference_executor = inference_executor
        self.tts_executor = tts_executor
        self.output_executor = output_executor or tts_executor

        self.state = StateMachine(initial=AssistantState.IDLE)

        # Internal split queue so STT can consume independently from watcher logic.
        self._stt_vad_q: asyncio.Queue = asyncio.Queue(maxsize=10)

        self._llm_task: asyncio.Task | None = None
        self._tts_task: asyncio.Task | None = None

        self._llm_cancel = threading.Event()
        self._tts_cancel = threading.Event()

        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    def _transition_safe(self, new_state: AssistantState) -> None:
        if self.state.current == new_state:
            return
        try:
            self.state.transition(new_state)
        except InvalidTransitionError:
            logger.debug(
                "Ignoring invalid transition %s -> %s",
                self.state.current.name,
                new_state.name,
            )

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        self._tasks = [
            asyncio.create_task(self._vad_dispatch_loop(), name="vad_dispatch"),
            asyncio.create_task(self._stt_to_llm_loop(), name="stt_to_llm"),
            asyncio.create_task(
                self.whisper_engine.run(
                    self._stt_vad_q,
                    self.stt_result_q,
                    self.inference_executor,
                    loop,
                ),
                name="stt_loop",
            ),
            asyncio.create_task(self._tts_playback_loop(), name="tts_loop"),
            asyncio.create_task(
                self.audio_output.run(self.output_executor),
                name="audio_output_loop",
            ),
        ]

        self._tts_task = self._tasks[3]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            await self._shutdown()
            raise

    async def _vad_dispatch_loop(self) -> None:
        while True:
            event = await self.vad_event_q.get()
            try:
                if isinstance(event, SpeechStartEvent):
                    if self.state.is_speaking():
                        await self._handle_interrupt(event)
                        continue
                    self._transition_safe(AssistantState.LISTENING)

                await self._stt_vad_q.put(event)
            finally:
                self.vad_event_q.task_done()

    async def _stt_to_llm_loop(self) -> None:
        while True:
            result = await self.stt_result_q.get()
            try:
                if not isinstance(result, TranscriptionResult):
                    continue

                text = result.text.strip()
                if not text:
                    self._transition_safe(AssistantState.IDLE)
                    continue

                self._transition_safe(AssistantState.TRANSCRIBING)
                self.conversation_manager.add_user_message(text)
                prompt = self.conversation_manager.format_prompt()

                self._transition_safe(AssistantState.THINKING)

                if self._llm_task is not None and not self._llm_task.done():
                    self._llm_task.cancel()

                self._llm_cancel.clear()
                self._llm_task = asyncio.create_task(self._llm_stream_loop(prompt))
            finally:
                self.stt_result_q.task_done()

    async def _llm_stream_loop(self, prompt: str) -> None:
        token_q: asyncio.Queue = asyncio.Queue(maxsize=256)
        producer_task = asyncio.create_task(
            self.llama_engine.stream(
                prompt,
                token_q,
                self._llm_cancel,
                self.inference_executor,
            )
        )

        full_response = ""
        sentence_buffer = ""
        speaking_started = False

        try:
            while True:
                token: LLMToken = await token_q.get()
                try:
                    if token.is_final:
                        if sentence_buffer.strip():
                            await self.sentence_q.put(SentenceReady(text=sentence_buffer.strip()))
                            if not speaking_started:
                                self._transition_safe(AssistantState.SPEAKING)
                                speaking_started = True
                            sentence_buffer = ""
                        break

                    txt = token.text
                    full_response += txt
                    sentence_buffer += txt

                    should_flush = (
                        bool(sentence_buffer)
                        and (
                            sentence_buffer[-1] in config.SENTENCE_FLUSH_CHARS
                            or len(sentence_buffer) >= config.SENTENCE_MAX_CHARS
                        )
                    )

                    if should_flush:
                        await self.sentence_q.put(SentenceReady(text=sentence_buffer.strip()))
                        if not speaking_started:
                            self._transition_safe(AssistantState.SPEAKING)
                            speaking_started = True
                        sentence_buffer = ""
                finally:
                    token_q.task_done()

            if not self._llm_cancel.is_set() and full_response.strip():
                self.conversation_manager.add_assistant_message(full_response.strip())

            self._transition_safe(AssistantState.IDLE)
        except asyncio.CancelledError:
            self._llm_cancel.set()
            raise
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    async def _tts_playback_loop(self) -> None:
        await self.piper_engine.run(
            self.sentence_q,
            self.audio_out_q,
            self.tts_executor,
            self._tts_cancel,
        )

    async def _handle_interrupt(self, event: SpeechStartEvent) -> None:
        self._transition_safe(AssistantState.INTERRUPTED)

        self._llm_cancel.set()
        self._tts_cancel.set()

        if self._llm_task is not None and not self._llm_task.done():
            self._llm_task.cancel()
            try:
                await self._llm_task
            except asyncio.CancelledError:
                pass

        # Drain pending sentence queue to prevent stale TTS after interruption.
        while not self.sentence_q.empty():
            try:
                self.sentence_q.get_nowait()
                self.sentence_q.task_done()
            except asyncio.QueueEmpty:
                break

        self.audio_output.flush()

        self._llm_cancel.clear()
        self._tts_cancel.clear()

        await self._stt_vad_q.put(event)
        self._transition_safe(AssistantState.LISTENING)

    async def _shutdown(self) -> None:
        if self._stopping:
            return
        self._stopping = True

        self._transition_safe(AssistantState.STOPPING)
        self._llm_cancel.set()
        self._tts_cancel.set()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        try:
            self.audio_output.stop()
        except Exception as exc:
            logger.debug("Audio output stop failed during shutdown: %s", exc)
