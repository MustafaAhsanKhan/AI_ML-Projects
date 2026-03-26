import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Generator

import config
from core.events import LLMToken

logger = logging.getLogger(__name__)


class LlamaEngine:
    """
    LLM engine backed by llama-cpp-python.

    Contracts:
    - Input prompt: full chat-formatted text
    - Output: LLMToken events, ending with is_final=True sentinel
    - Cancellation: checked between streamed tokens using threading.Event
    """

    def __init__(self, model_path: Path) -> None:
        self._model_path = Path(model_path)
        self._model: Any | None = None

    def load(self) -> None:
        """Load GGUF model using llama-cpp-python."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"LLM model not found: {self._model_path}")

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed in this environment. "
                "Install it with Metal support before running Phase 5 live tests."
            ) from exc

        logger.info("Loading LLM model from %s", self._model_path)
        self._model = Llama(
            model_path=str(self._model_path),
            n_gpu_layers=config.LLM_N_GPU_LAYERS,
            n_ctx=config.LLM_CONTEXT_SIZE,
            verbose=False,
        )
        logger.info("LLM model ready")

    def _extract_token_text(self, chunk: Any) -> str:
        """Extract token text from llama-cpp streaming chunk structure."""
        if isinstance(chunk, dict):
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices:
                choice0 = choices[0]
                if isinstance(choice0, dict):
                    if "text" in choice0:
                        return str(choice0.get("text") or "")

                    delta = choice0.get("delta")
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if content is not None:
                            return str(content)
        return ""

    def _stream_sync(
        self,
        prompt: str,
        cancel_event: threading.Event,
    ) -> Generator[str, None, None]:
        """Synchronous token stream for executor thread."""
        if self._model is None:
            raise RuntimeError("LlamaEngine.load() must be called before _stream_sync()")

        stream_iter = self._model(
            prompt,
            stream=True,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        )

        for chunk in stream_iter:
            if cancel_event.is_set():
                logger.debug("LLM stream cancelled")
                return

            token = self._extract_token_text(chunk)
            if token:
                yield token

    async def stream(
        self,
        prompt: str,
        token_q: asyncio.Queue,
        cancel_event: threading.Event,
        executor: ThreadPoolExecutor,
    ) -> None:
        """
        Push streamed tokens into token_q and always finalize with sentinel token.
        """
        if self._model is None:
            self.load()

        loop = asyncio.get_running_loop()

        def _producer() -> None:
            for token in self._stream_sync(prompt, cancel_event):
                fut = asyncio.run_coroutine_threadsafe(
                    token_q.put(LLMToken(text=token, is_final=False)),
                    loop,
                )
                fut.result()

        try:
            await loop.run_in_executor(executor, _producer)
        finally:
            await token_q.put(LLMToken(text="", is_final=True))
