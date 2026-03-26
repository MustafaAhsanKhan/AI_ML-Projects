import asyncio
import io
import logging
import math
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

import config
from core.events import AudioChunk, SentenceReady

logger = logging.getLogger(__name__)


class PiperEngine:
    """
    TTS engine wrapper around Piper.

    Output contract:
    - AudioChunk.pcm is float32 mono at TTS_SAMPLE_RATE.
    """

    def __init__(self, model_path: Path, config_path: Path) -> None:
        self._model_path = Path(model_path)
        self._config_path = Path(config_path)
        self._voice: Any | None = None

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {self._model_path}")
        if not self._config_path.exists():
            raise FileNotFoundError(f"Piper config not found: {self._config_path}")

        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise RuntimeError("piper-tts is not installed") from exc

        logger.info("Loading Piper voice from %s", self._model_path)
        self._voice = PiperVoice.load(str(self._model_path), str(self._config_path))
        logger.info("Piper voice ready")

    def _bytes_to_float32_pcm(self, raw_pcm: bytes) -> np.ndarray:
        if not raw_pcm:
            return np.array([], dtype=np.float32)

        count = len(raw_pcm) // 2
        if count == 0:
            return np.array([], dtype=np.float32)

        samples = struct.unpack("<" + "h" * count, raw_pcm[: count * 2])
        return (np.asarray(samples, dtype=np.float32) / 32768.0).astype(np.float32)

    def _resample_if_needed(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == config.TTS_SAMPLE_RATE:
            return audio.astype(np.float32, copy=False)

        g = math.gcd(source_rate, config.TTS_SAMPLE_RATE)
        up = config.TTS_SAMPLE_RATE // g
        down = source_rate // g
        return resample_poly(audio, up, down).astype(np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        if self._voice is None:
            raise RuntimeError("PiperEngine.load() must be called before synthesize()")

        text = text.strip()
        if not text:
            return np.array([], dtype=np.float32)

        buffer = io.BytesIO()

        if hasattr(self._voice, "synthesize_stream_raw"):
            self._voice.synthesize_stream_raw(text, buffer)
            audio = self._bytes_to_float32_pcm(buffer.getvalue())
        elif hasattr(self._voice, "synthesize"):
            audio = np.asarray(self._voice.synthesize(text), dtype=np.float32).reshape(-1)
        else:
            raise RuntimeError("Unsupported PiperVoice API")

        source_rate = int(getattr(self._voice, "sample_rate", config.TTS_SAMPLE_RATE))
        return self._resample_if_needed(audio, source_rate)

    def chunk_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        audio = audio.reshape(-1).astype(np.float32, copy=False)
        if len(audio) == 0:
            return []

        chunk_size = config.OUTPUT_CHUNK_SIZE
        out: list[np.ndarray] = []

        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            if len(chunk) < chunk_size:
                padded = np.zeros(chunk_size, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded
            out.append(chunk)

        return out

    async def run(
        self,
        sentence_q: asyncio.Queue,
        audio_out_q: asyncio.Queue,
        executor: ThreadPoolExecutor,
        cancel_event: threading.Event,
    ) -> None:
        if self._voice is None:
            self.load()

        loop = asyncio.get_running_loop()
        logger.info("Piper TTS loop started")

        try:
            while True:
                item = await sentence_q.get()
                try:
                    if not isinstance(item, SentenceReady):
                        continue
                    if cancel_event.is_set():
                        continue

                    audio = await loop.run_in_executor(executor, self.synthesize, item.text)
                    if cancel_event.is_set():
                        continue

                    for chunk in self.chunk_audio(audio):
                        if cancel_event.is_set():
                            break
                        await audio_out_q.put(AudioChunk(pcm=chunk))
                finally:
                    sentence_q.task_done()
        except asyncio.CancelledError:
            logger.info("Piper TTS loop cancelled")
            raise
