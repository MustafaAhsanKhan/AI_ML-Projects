import streamlit as st
import requests

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b-instruct-q4_0"

st.title("Local Personal Assistant")

# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("Ask me anything..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send request to Ollama
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    reply = response.json()["response"]

    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)