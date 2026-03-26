import logging

import config

logger = logging.getLogger(__name__)


class ConversationManager:
    """Maintains conversation history and formats Llama 3 style prompts."""

    def __init__(self, system_prompt: str, max_tokens: int) -> None:
        self._system_prompt = system_prompt.strip()
        self._max_tokens = max_tokens
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt}
        ]

    def add_user_message(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.messages.append({"role": "user", "content": text})
        self._trim_to_context()

    def add_assistant_message(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.messages.append({"role": "assistant", "content": text})
        self._trim_to_context()

    def _estimate_tokens(self) -> int:
        # Coarse estimate from plan: ~4 chars per token plus per-message overhead.
        total = 0
        for msg in self.messages:
            total += len(msg["content"]) // 4
            total += 8
        return total

    def _trim_to_context(self) -> None:
        # Keep system prompt pinned at index 0.
        while len(self.messages) > 1 and self._estimate_tokens() > self._max_tokens:
            removed = self.messages.pop(1)
            logger.debug(
                "Trimmed conversation message role=%s chars=%d",
                removed.get("role", "unknown"),
                len(removed.get("content", "")),
            )

    def format_prompt(self) -> str:
        self._trim_to_context()

        parts = ["<|begin_of_text|>"]
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"].strip()
            parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )

        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self._system_prompt}]


def default_conversation_manager() -> ConversationManager:
    return ConversationManager(config.SYSTEM_PROMPT, config.LLM_CONTEXT_SIZE)
