import logging
from collections import deque

import numpy as np

import config

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    Accumulates audio chunks during an utterance and prepends a pre-roll window
    so that the very start of speech (before VAD onset) is not clipped.

    Lifecycle per utterance:
        add_chunk() called continuously by the VAD loop
        start_recording() called on SpeechStartEvent
        add_chunk() continues, now also appending to the recording buffer
        stop_recording() called on SpeechEndEvent → returns full waveform
        reset() called on barge-in to discard a partial recording
    """

    def __init__(self, pre_roll_chunks: int = config.VAD_PRE_ROLL_CHUNKS) -> None:
        self._pre_roll_chunks = pre_roll_chunks
        # Rolling window of recent chunks — always maintained, recording or not
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)
        # Active recording buffer (list of numpy arrays, concatenated on stop)
        self._buffer: list[np.ndarray] = []
        self._recording: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: np.ndarray) -> None:
        """
        Accept a new audio chunk from the input stream.

        Always updates the pre-roll window. When recording is active,
        also appends to the utterance buffer.

        Args:
            chunk: float32 mono array of BLOCK_SIZE samples.
        """
        self._pre_roll.append(chunk.copy())
        if self._recording:
            self._buffer.append(chunk.copy())

    def start_recording(self) -> None:
        """
        Mark the start of an utterance.

        The current pre-roll window is seeded into the buffer so that
        audio immediately before VAD onset is included in the output.
        """
        if self._recording:
            logger.warning("start_recording called while already recording — resetting")
            self._buffer.clear()

        # Seed buffer with pre-roll so we don't clip the utterance start
        self._buffer = [chunk.copy() for chunk in self._pre_roll]
        self._recording = True
        logger.debug(
            "Recording started — pre-roll seeded with %d chunks (~%dms)",
            len(self._buffer),
            len(self._buffer) * config.BLOCK_SIZE * 1000 // config.SAMPLE_RATE,
        )

    def stop_recording(self) -> np.ndarray:
        """
        End the utterance and return the complete waveform.

        Returns:
            float32 mono numpy array at SAMPLE_RATE containing the full
            utterance including pre-roll.  Returns an empty array if
            stop_recording is called without a matching start_recording.
        """
        if not self._recording:
            logger.warning("stop_recording called while not recording")
            return np.array([], dtype=np.float32)

        self._recording = False

        if not self._buffer:
            self._buffer.clear()
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self._buffer, axis=0).astype(np.float32)
        duration_ms = len(audio) * 1000 // config.SAMPLE_RATE
        logger.debug("Recording stopped — %d samples (%dms)", len(audio), duration_ms)
        self._buffer.clear()
        return audio

    def reset(self) -> None:
        """
        Discard any in-progress recording without returning data.
        Used during barge-in to cleanly abort the current utterance.
        The pre-roll window is intentionally preserved so the next
        start_recording() inherits recent audio context.
        """
        self._recording = False
        self._buffer.clear()
        logger.debug("AudioBuffer reset (barge-in)")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def buffered_seconds(self) -> float:
        """Approximate duration of audio currently in the recording buffer."""
        if not self._buffer:
            return 0.0
        total_samples = sum(len(c) for c in self._buffer)
        return total_samples / config.SAMPLE_RATE
