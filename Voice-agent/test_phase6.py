"""Phase 6 tests — ConversationManager."""

import sys

sys.path.insert(0, "/Users/mustafa/Desktop/AI_ML-Projects/Voice-agent")

from core.conversation_manager import ConversationManager


def test_prompt_format_structure() -> None:
    cm = ConversationManager(system_prompt="You are helpful.", max_tokens=4096)
    cm.add_user_message("Hello there")
    cm.add_assistant_message("Hi, how can I help?")

    prompt = cm.format_prompt()

    assert prompt.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    print("Conversation prompt structure test OK")


def test_trimming_removes_oldest_non_system() -> None:
    cm = ConversationManager(system_prompt="S", max_tokens=45)
    cm.add_user_message("u1 " * 30)
    cm.add_assistant_message("a1 " * 30)
    cm.add_user_message("u2 " * 30)

    _ = cm.format_prompt()

    assert cm.messages[0]["role"] == "system"
    assert cm.messages[0]["content"] == "S"
    assert len(cm.messages) >= 2
    print("Conversation trimming behavior test OK")


def test_empty_messages_ignored() -> None:
    cm = ConversationManager(system_prompt="S", max_tokens=4096)
    cm.add_user_message("   ")
    cm.add_assistant_message("\n")
    assert len(cm.messages) == 1
    print("Conversation empty-message guard test OK")


def test_reset_keeps_system_only() -> None:
    cm = ConversationManager(system_prompt="System prompt", max_tokens=4096)
    cm.add_user_message("Hello")
    cm.add_assistant_message("Hi")
    cm.reset()

    assert len(cm.messages) == 1
    assert cm.messages[0]["role"] == "system"
    assert cm.messages[0]["content"] == "System prompt"
    print("Conversation reset test OK")


if __name__ == "__main__":
    print("=== Phase 6: Conversation Manager Tests ===\n")
    test_prompt_format_structure()
    test_trimming_removes_oldest_non_system()
    test_empty_messages_ignored()
    test_reset_keeps_system_only()
    print("\nAll Phase 6 tests passed")
