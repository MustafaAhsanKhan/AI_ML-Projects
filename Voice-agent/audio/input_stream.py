import asyncio
import logging

import numpy as np
import sounddevice as sd

import config

logger = logging.getLogger(__name__)


class AudioInputStream:
    """
    Captures microphone audio via sounddevice and pushes float32 mono chunks
    into an asyncio.Queue for downstream processing.

    The sounddevice callback fires in a C-level thread.  All queue puts are
    routed through loop.call_soon_threadsafe so they are safe to consume from
    the asyncio event loop without any locking.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._queue = queue
        self._loop = loop
        self._stream: sd.InputStream | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the input stream and begin capturing audio."""
        if self._stream is not None and self._stream.active:
            logger.warning("AudioInputStream already running")
            return

        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=config.BLOCK_SIZE,
            channels=config.CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Microphone open — %dHz, block=%d samples (~%dms)",
            config.SAMPLE_RATE,
            config.BLOCK_SIZE,
            config.BLOCK_SIZE * 1000 // config.SAMPLE_RATE,
        )

    def stop(self) -> None:
        """Stop the input stream and release the microphone."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Microphone closed")

    # ------------------------------------------------------------------
    # sounddevice callback (runs in C thread)
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,   # shape (blocksize, channels), float32
        frames: int,
        time,                 # CData time info — unused
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            # Log overflows / drops without blocking the callback thread
            logger.debug("Input stream status: %s", status)

        # Flatten to 1-D mono array and copy (sounddevice reuses the buffer)
        chunk: np.ndarray = indata[:, 0].copy()

        # Bridge C thread → asyncio event loop thread-safely.
        # put_nowait is used so that a backed-up queue causes dropped frames
        # rather than blocking the audio callback (which would cause glitches).
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
        except asyncio.QueueFull:
            logger.warning("audio_in_q full — dropping chunk (consumer too slow)")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active
