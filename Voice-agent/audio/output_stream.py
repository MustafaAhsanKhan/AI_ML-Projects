import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd

import config
from core.events import AudioChunk

logger = logging.getLogger(__name__)


class AudioOutputStream:
    """
    Consumes AudioChunk objects from an asyncio.Queue and plays them through
    the system speaker via sounddevice.

    The output stream runs at TTS_SAMPLE_RATE (Piper's native rate) which is
    independent of the capture rate used by the input stream.

    Stream writes are offloaded to a ThreadPoolExecutor so that a momentarily
    full device buffer cannot stall the asyncio event loop.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._queue = queue
        self._loop = loop
        self._stream: sd.OutputStream | None = None
        self._active: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the output stream, ready for playback."""
        if self._stream is not None and self._stream.active:
            logger.warning("AudioOutputStream already running")
            return

        self._stream = sd.OutputStream(
            samplerate=config.TTS_SAMPLE_RATE,
            blocksize=config.OUTPUT_CHUNK_SIZE,
            channels=config.CHANNELS,
            dtype="float32",
        )
        self._stream.start()
        self._active = True
        logger.info(
            "Speaker open — %dHz, chunk=%d samples (~%dms)",
            config.TTS_SAMPLE_RATE,
            config.OUTPUT_CHUNK_SIZE,
            config.OUTPUT_CHUNK_SIZE * 1000 // config.TTS_SAMPLE_RATE,
        )

    def stop(self) -> None:
        """Close the output stream."""
        self._active = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Speaker closed")

    def flush(self) -> None:
        """
        Immediately discard all pending audio — used during barge-in.

        Drains the queue and stops/restarts the device stream to purge
        whatever is already buffered at the hardware level.  The stream
        is left open and ready for the next utterance.
        """
        # Drain the asyncio queue (best-effort, non-blocking)
        drained = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                drained += 1
            except asyncio.QueueEmpty:
                break

        # Stop and restart the device stream to clear the hardware ring buffer
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.start()
            except Exception as exc:
                logger.warning("Error restarting output stream during flush: %s", exc)

        logger.debug("Output flushed — %d chunks discarded", drained)

    # ------------------------------------------------------------------
    # Async run loop
    # ------------------------------------------------------------------

    async def run(self, executor: ThreadPoolExecutor) -> None:
        """
        Main playback coroutine.  Dequeues AudioChunk objects and writes
        their PCM data to the sounddevice output stream.

        Blocks (in the executor) only as long as the device buffer needs
        draining, which is normally sub-millisecond for 1024-sample chunks.

        This coroutine runs for the lifetime of the application and is
        cancelled by the orchestrator on shutdown.
        """
        if self._stream is None or not self._stream.active:
            raise RuntimeError("AudioOutputStream.start() must be called before run()")

        logger.debug("Audio output loop started")
        try:
            while True:
                chunk: AudioChunk = await self._queue.get()
                pcm = chunk.pcm.astype(np.float32)

                # sounddevice.write expects shape (frames, channels)
                pcm_2d = pcm.reshape(-1, 1)

                try:
                    await self._loop.run_in_executor(
                        executor, self._stream.write, pcm_2d
                    )
                except Exception as exc:
                    logger.warning("Output stream write error: %s", exc)
                finally:
                    self._queue.task_done()

        except asyncio.CancelledError:
            logger.debug("Audio output loop cancelled")
            raise

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        """True when there are chunks queued or being written."""
        return not self._queue.empty()

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active
