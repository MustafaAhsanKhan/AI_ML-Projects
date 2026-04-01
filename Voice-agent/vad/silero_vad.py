import asyncio
import logging
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

import config
from audio.audio_buffer import AudioBuffer
from core.events import SpeechEndEvent, SpeechStartEvent

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD wrapper that consumes 16kHz audio chunks and emits speech
    boundary events for downstream STT.

    Contracts:
    - Input chunks: float32 mono, BLOCK_SIZE samples at SAMPLE_RATE (16kHz)
    - SpeechStartEvent.pre_roll_audio: float32 mono, 16kHz
    - SpeechEndEvent.audio: float32 mono, 16kHz
    """

    def __init__(self, model_path: Path) -> None:
        self._model_path = Path(model_path)
        self._sample_rate = config.SAMPLE_RATE
        self._threshold = config.VAD_THRESHOLD
        self._silence_chunks = config.VAD_SILENCE_CHUNKS

        self._session: ort.InferenceSession | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

        self._h: np.ndarray | None = None
        self._c: np.ndarray | None = None
        self._state: np.ndarray | None = None

        self._pre_roll: deque[np.ndarray] = deque(maxlen=config.VAD_PRE_ROLL_CHUNKS)
        self._buffer = AudioBuffer(pre_roll_chunks=config.VAD_PRE_ROLL_CHUNKS)

    def load(self) -> None:
        """Load ONNX model and initialize recurrent state."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {self._model_path}")

        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self._model_path), providers=providers)
        self._input_names = [meta.name for meta in self._session.get_inputs()]
        self._output_names = [meta.name for meta in self._session.get_outputs()]

        self._initialize_recurrent_state()

        logger.info(
            "Silero VAD loaded: inputs=%s outputs=%s",
            self._input_names,
            self._output_names,
        )

    def _initialize_recurrent_state(self) -> None:
        # Handle both common Silero ONNX signatures:
        # 1) split recurrent inputs: h/c
        # 2) single recurrent input: state
        has_state = any(name.lower() == "state" for name in self._input_names)
        if has_state:
            state_shape = self._get_input_shape_hint("state", default=[2, 1, 128])
            self._state = np.zeros(state_shape, dtype=np.float32)
            self._h = None
            self._c = None
            return

        # Fallback to split-state variants.
        h_shape = self._get_input_shape_hint("h", default=[2, 1, 64])
        c_shape = self._get_input_shape_hint("c", default=[2, 1, 64])

        self._h = np.zeros(h_shape, dtype=np.float32)
        self._c = np.zeros(c_shape, dtype=np.float32)
        self._state = None

    def _get_input_shape_hint(self, token: str, default: list[int]) -> list[int]:
        if self._session is None:
            return default

        for meta in self._session.get_inputs():
            if token in meta.name.lower():
                shape: list[int] = []
                for dim in meta.shape:
                    if isinstance(dim, int) and dim > 0:
                        shape.append(dim)
                    else:
                        # Dynamic dims get mapped to a safe singleton.
                        shape.append(1)
                return shape if shape else default
        return default

    def process_chunk(self, chunk: np.ndarray) -> float:
        """Return speech probability in [0, 1] for one audio chunk."""
        if self._session is None:
            raise RuntimeError("SileroVAD.load() must be called before process_chunk()")

        if chunk.ndim != 1:
            chunk = chunk.reshape(-1)

        x = chunk.astype(np.float32, copy=False)
        x = np.ascontiguousarray(x).reshape(1, -1)

        feed: dict[str, np.ndarray] = {}
        for name in self._input_names:
            lname = name.lower()
            if lname in {"input", "x", "audio"} or "input" in lname:
                feed[name] = x
            elif lname in {"sr", "sample_rate"} or "sr" in lname:
                feed[name] = np.array(self._sample_rate, dtype=np.int64)
            elif lname.startswith("h") or lname == "h":
                if self._h is None:
                    self._initialize_recurrent_state()
                feed[name] = self._h
            elif lname.startswith("c") or lname == "c":
                if self._c is None:
                    self._initialize_recurrent_state()
                feed[name] = self._c
            elif lname == "state" or lname.startswith("state"):
                if self._state is None:
                    self._initialize_recurrent_state()
                feed[name] = self._state
            else:
                # Best-effort fallback for uncommon input names.
                if "state" in lname:
                    if self._state is not None:
                        feed[name] = self._state
                    elif self._h is not None:
                        feed[name] = self._h

        outputs = self._session.run(None, feed)

        # Map outputs by name when available.
        out_map = {
            self._output_names[i]: outputs[i]
            for i in range(min(len(self._output_names), len(outputs)))
        }

        # Update recurrent state when provided.
        for name, arr in out_map.items():
            lname = name.lower()
            if lname.startswith("h") or lname in {"hn", "h_out"}:
                self._h = np.asarray(arr, dtype=np.float32)
            elif lname.startswith("c") or lname in {"cn", "c_out"}:
                self._c = np.asarray(arr, dtype=np.float32)
            elif lname == "staten" or lname.startswith("state"):
                self._state = np.asarray(arr, dtype=np.float32)

        # Fallback positional state update for common [prob, h, c] signatures.
        if len(outputs) >= 3:
            self._h = np.asarray(outputs[1], dtype=np.float32)
            self._c = np.asarray(outputs[2], dtype=np.float32)

        prob_array: np.ndarray
        if out_map:
            # Prefer explicitly named probability outputs when present.
            candidate = None
            for key in ("output", "speech_prob", "prob", "probability"):
                if key in out_map:
                    candidate = out_map[key]
                    break
            if candidate is None:
                candidate = next(iter(out_map.values()))
            prob_array = np.asarray(candidate)
        else:
            prob_array = np.asarray(outputs[0])

        prob = float(prob_array.reshape(-1)[0])
        return max(0.0, min(1.0, prob))

    async def run(
        self,
        audio_in_q: asyncio.Queue,
        vad_event_q: asyncio.Queue,
    ) -> None:
        """Consume audio chunks and emit SpeechStartEvent/SpeechEndEvent."""
        if self._session is None:
            self.load()

        speaking = False
        silence_count = 0

        logger.info("Silero VAD loop started")

        try:
            while True:
                chunk: np.ndarray = await audio_in_q.get()
                try:
                    chunk = chunk.astype(np.float32, copy=False).reshape(-1)

                    # Keep local pre-roll for SpeechStartEvent payload.
                    self._pre_roll.append(chunk.copy())

                    # AudioBuffer always tracks pre-roll and active recording.
                    self._buffer.add_chunk(chunk)

                    prob = self.process_chunk(chunk)

                    if not speaking and prob > self._threshold:
                        speaking = True
                        silence_count = 0
                        self._buffer.start_recording()

                        if self._pre_roll:
                            pre_roll_audio = np.concatenate(list(self._pre_roll), axis=0).astype(
                                np.float32
                            )
                        else:
                            pre_roll_audio = np.array([], dtype=np.float32)

                        await vad_event_q.put(SpeechStartEvent(pre_roll_audio=pre_roll_audio))
                        logger.info("Speech start detected (prob=%.3f)", prob)

                    elif speaking:
                        if prob > self._threshold:
                            silence_count = 0
                        else:
                            silence_count += 1

                        if silence_count >= self._silence_chunks:
                            audio = self._buffer.stop_recording()
                            speaking = False
                            silence_count = 0

                            if len(audio) > 0:
                                await vad_event_q.put(SpeechEndEvent(audio=audio))
                                logger.info(
                                    "Speech end detected (%d samples)",
                                    len(audio),
                                )
                finally:
                    audio_in_q.task_done()

        except asyncio.CancelledError:
            logger.info("Silero VAD loop cancelled")
            raise
