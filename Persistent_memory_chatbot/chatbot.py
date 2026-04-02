import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def get_response(messages):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3",
            "messages": messages,
            "stream": False
        }
    )

    return response.json()["message"]["content"]