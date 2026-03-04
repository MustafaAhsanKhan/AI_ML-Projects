# Local Personal Assistant Chatbot

This project implements a locally-running personal assistant chatbot using Ollama and Streamlit. It runs entirely on your machine — no cloud API keys required.

## Features

- **Fully Local**: Runs on your own hardware using Ollama, with no data sent to external servers
- **Streamlit UI**: Clean, browser-based chat interface with persistent conversation history
- **Session Memory**: Chat history is preserved throughout your session
- **Customizable Model**: Easily swap out the LLM by changing a single variable

## How It Works

The project uses:
- `streamlit` for the web-based chat interface
- `requests` to communicate with the local Ollama API
- `ollama` as the local LLM runtime (runs models like LLaMA 3 on your machine)
- Session state to maintain chat history across interactions

## Requirements

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/local-run-chatbot.git
   cd local-run-chatbot
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Pull the model with Ollama:
   ```bash
   ollama pull llama3:8b-instruct-q4_0
   ```

## Usage

1. Make sure Ollama is running in the background:
   ```bash
   ollama serve
   ```

2. Activate your virtual environment if not already active.

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`.

5. Start chatting — type any question in the input box and press Enter.