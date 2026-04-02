from fastapi import FastAPI
from models import ChatRequest
from chatbot import get_response
from memory import (
    EARLY_SUMMARY_CHAR_THRESHOLD,
    RECENT_MESSAGE_WINDOW,
    SUMMARY_EVERY_MESSAGES,
    append_message,
    build_model_context,
    update_summary,
)

app = FastAPI()

@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id

    # Add user message
    append_message(session_id, "user", req.message)

    # Refresh summary first if older messages crossed the threshold.
    update_summary(session_id)

    # Send summary + recent raw messages to the model.
    context_messages = build_model_context(session_id)

    # Get AI response
    response = get_response(context_messages)

    # Store assistant response
    append_message(session_id, "assistant", response)

    # Refresh summary again after full user-assistant turn.
    summary = update_summary(session_id)

    return {
        "response": response,
        "summary": summary,
        "summary_policy": {
            "message_interval": SUMMARY_EVERY_MESSAGES,
            "recent_message_window": RECENT_MESSAGE_WINDOW,
            "early_char_threshold": EARLY_SUMMARY_CHAR_THRESHOLD,
        },
    }