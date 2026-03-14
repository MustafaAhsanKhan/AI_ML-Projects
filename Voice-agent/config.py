from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

MODELS_DIR = BASE_DIR / "models"

# Set this to the absolute path of your Llama 3 8B Instruct Q4_0 .gguf file.
# Example: Path("/Users/you/.cache/lm-studio/models/llama-3-8b-instruct.Q4_0.gguf")
LLM_MODEL_PATH: Path = Path("/your/actual/path/to/Meta-Llama-3-8B-Instruct-Q4_0.gguf")

WHISPER_MODEL_PATH: Path = MODELS_DIR / "whisper"
PIPER_MODEL_PATH: Path = MODELS_DIR / "piper" / "en_US-lessac-medium.onnx"
PIPER_CONFIG_PATH: Path = MODELS_DIR / "piper" / "en_US-lessac-medium.onnx.json"
SILERO_MODEL_PATH: Path = MODELS_DIR / "silero" / "silero_vad.onnx"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000        # Hz — Whisper + Silero native rate
BLOCK_SIZE: int = 512           # samples per chunk (~32ms at 16kHz)
CHANNELS: int = 1

# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

VAD_THRESHOLD: float = 0.5      # Speech onset probability
VAD_SILENCE_CHUNKS: int = 20    # Consecutive silent chunks before SpeechEnd (~640ms)
VAD_PRE_ROLL_CHUNKS: int = 5    # Chunks prepended before speech onset (~160ms)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_N_GPU_LAYERS: int = -1      # -1 = offload all layers to Metal
LLM_CONTEXT_SIZE: int = 4096
LLM_MAX_TOKENS: int = 512
LLM_TEMPERATURE: float = 0.7

# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE: str = "base.en"

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

TTS_SAMPLE_RATE: int = 22050    # Piper native output sample rate
OUTPUT_CHUNK_SIZE: int = 1024   # PCM samples per audio_out_q chunk

# ---------------------------------------------------------------------------
# Sentence buffer (LLM → TTS boundary)
# ---------------------------------------------------------------------------

SENTENCE_MAX_CHARS: int = 200   # Force-flush if buffer exceeds this
SENTENCE_FLUSH_CHARS: str = ".?!\n"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a helpful voice assistant. "
    "Give clear, concise responses suitable for spoken conversation. "
    "Avoid markdown, bullet points, or any formatting — speak in plain sentences only."
)
