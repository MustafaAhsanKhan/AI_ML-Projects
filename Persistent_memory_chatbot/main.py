from fastapi import FastAPI
from models import ChatRequest
from chatbot import get_response
from memory import get_history, append_message

app = FastAPI()

@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id

    # Add user message
    append_message(session_id, "user", req.message)

    # Get full conversation
    history = get_history(session_id)

    # Get AI response
    response = get_response(history)

    # Store assistant response
    append_message(session_id, "assistant", response)

    return {"response": response}