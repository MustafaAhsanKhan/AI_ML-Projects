import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from core.events import SpeechEndEvent, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperEngine:
    """
    Speech-to-text engine using faster-whisper.

    Contracts:
    - Input audio: float32 mono at 16kHz (from SpeechEndEvent)
    - Output: TranscriptionResult(text=...)
    """

    def __init__(self, model_size: str, model_path: Path) -> None:
        self._model_size = model_size
        self._model_path = Path(model_path)
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Instantiate the faster-whisper model."""
        # Prefer explicit local path if provided; otherwise use model size name.
        model_ref = str(self._model_path) if self._model_path.exists() else self._model_size

        logger.info(
            "Loading Whisper model (%s)",
            model_ref,
        )

        self._model = WhisperModel(
            model_ref,
            device="auto",
            compute_type="int8",
        )

        logger.info("Whisper model ready")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe one audio segment to plain text."""
        if self._model is None:
            raise RuntimeError("WhisperEngine.load() must be called before transcribe()")

        if audio.ndim != 1:
            audio = audio.reshape(-1)

        audio = audio.astype(np.float32, copy=False)

        segments, _info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=False,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text

    async def run(
        self,
        vad_event_q: asyncio.Queue,
        stt_result_q: asyncio.Queue,
        executor: ThreadPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Consume VAD events and produce transcription results.

        Only SpeechEndEvent triggers transcription; all other event types are ignored.
        """
        if self._model is None:
            self.load()

        logger.info("Whisper STT loop started")

        try:
            while True:
                event = await vad_event_q.get()
                try:
                    if not isinstance(event, SpeechEndEvent):
                        continue

                    audio = event.audio.astype(np.float32, copy=False)
                    if len(audio) == 0:
                        logger.debug("Skipping empty SpeechEndEvent audio")
                        continue

                    text = await loop.run_in_executor(executor, self.transcribe, audio)
                    text = text.strip()

                    if not text:
                        logger.debug("Skipping empty transcription")
                        continue

                    await stt_result_q.put(TranscriptionResult(text=text))
                    logger.debug("Transcription emitted (%d chars)", len(text))
                finally:
                    vad_event_q.task_done()

        except asyncio.CancelledError:
            logger.info("Whisper STT loop cancelled")
            raise
