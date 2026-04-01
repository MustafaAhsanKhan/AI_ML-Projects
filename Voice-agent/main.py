import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import config
from audio.input_stream import AudioInputStream
from audio.output_stream import AudioOutputStream
from core.conversation_manager import ConversationManager
from core.realtime_orchestrator import RealtimeOrchestrator
from llm.llama_engine import LlamaEngine
from stt.whisper_engine import WhisperEngine
from tts.piper_engine import PiperEngine
from vad.silero_vad import SileroVAD


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def _mic_info_loop(audio_input: AudioInputStream, audio_in_q: asyncio.Queue) -> None:
    """Log basic microphone signal diagnostics every 0.5 seconds."""
    while True:
        await asyncio.sleep(0.5)
        rms, peak, _last_ts, chunks_seen = audio_input.get_metrics()

        logger.info(
            "Mic info | rms=%.5f peak=%.5f chunks=%d qsize=%d active=%s",
            rms,
            peak,
            chunks_seen,
            audio_in_q.qsize(),
            audio_input.is_active,
        )


async def main() -> None:
    _configure_logging()

    logger.info("Initializing queues")
    audio_in_q: asyncio.Queue = asyncio.Queue(maxsize=50)
    vad_event_q: asyncio.Queue = asyncio.Queue(maxsize=10)
    stt_result_q: asyncio.Queue = asyncio.Queue(maxsize=5)
    sentence_q: asyncio.Queue = asyncio.Queue(maxsize=10)
    audio_out_q: asyncio.Queue = asyncio.Queue(maxsize=100)

    logger.info("Initializing executors")
    inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
    output_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio_out")

    loop = asyncio.get_running_loop()

    logger.info("Initializing engines")
    vad = SileroVAD(config.SILERO_MODEL_PATH)
    whisper = WhisperEngine(config.WHISPER_MODEL_SIZE, config.WHISPER_MODEL_PATH)
    llama = LlamaEngine(config.LLM_MODEL_PATH)
    piper = PiperEngine(config.PIPER_MODEL_PATH, config.PIPER_CONFIG_PATH)

    logger.info("Loading VAD model")
    vad.load()
    logger.info("Loading Whisper model")
    whisper.load()
    logger.info("Loading LLM model")
    llama.load()
    logger.info("Loading Piper voice")
    piper.load()

    conversation_manager = ConversationManager(
        system_prompt=config.SYSTEM_PROMPT,
        max_tokens=config.LLM_CONTEXT_SIZE,
    )

    audio_input = AudioInputStream(queue=audio_in_q, loop=loop)
    audio_output = AudioOutputStream(queue=audio_out_q, loop=loop)

    orchestrator = RealtimeOrchestrator(
        whisper_engine=whisper,
        llama_engine=llama,
        piper_engine=piper,
        conversation_manager=conversation_manager,
        audio_output=audio_output,
        vad_event_q=vad_event_q,
        stt_result_q=stt_result_q,
        sentence_q=sentence_q,
        audio_out_q=audio_out_q,
        inference_executor=inference_executor,
        tts_executor=tts_executor,
        output_executor=output_executor,
    )

    vad_task: asyncio.Task | None = None
    orchestrator_task: asyncio.Task | None = None
    mic_info_task: asyncio.Task | None = None

    try:
        logger.info("Starting audio streams")
        audio_output.start()
        audio_input.start()

        vad_task = asyncio.create_task(vad.run(audio_in_q, vad_event_q), name="vad_loop")
        orchestrator_task = asyncio.create_task(orchestrator.run(), name="orchestrator")
        mic_info_task = asyncio.create_task(
            _mic_info_loop(audio_input, audio_in_q),
            name="mic_info_loop",
        )

        logger.info("Voice assistant running. Press Ctrl+C to stop.")
        await asyncio.gather(vad_task, orchestrator_task)
    finally:
        logger.info("Shutting down")

        if mic_info_task is not None and not mic_info_task.done():
            mic_info_task.cancel()
            await asyncio.gather(mic_info_task, return_exceptions=True)

        if vad_task is not None and not vad_task.done():
            vad_task.cancel()
            await asyncio.gather(vad_task, return_exceptions=True)

        if orchestrator_task is not None and not orchestrator_task.done():
            orchestrator_task.cancel()
            await asyncio.gather(orchestrator_task, return_exceptions=True)

        await orchestrator._shutdown()

        audio_input.stop()
        audio_output.stop()

        inference_executor.shutdown(wait=False, cancel_futures=True)
        tts_executor.shutdown(wait=False, cancel_futures=True)
        output_executor.shutdown(wait=False, cancel_futures=True)


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    run()
