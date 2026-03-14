# Local Realtime Voice Assistant — Full Project Plan

> Reference document for architecture decisions, component design, and implementation tasks.
> Do not delete. Updated as implementation progresses.

---

## Hardware Target

- MacBook Air M4, 16GB RAM, ~8GB available for models
- Metal GPU acceleration via llama.cpp + CoreML for faster-whisper

---

## Technology Stack

| Role | Library |
|---|---|
| LLM | `llama-cpp-python` (Metal build) |
| STT | `faster-whisper` (CoreML) |
| VAD | Silero VAD v5 via `onnxruntime` |
| TTS | `piper-tts` |
| Audio I/O | `sounddevice` |
| Concurrency | `asyncio` + `ThreadPoolExecutor` |

---

## Model Recommendations

| Component | Model | Size |
|---|---|---|
| LLM | Llama 3 8B Instruct Q4_0 | ~4.7 GB |
| STT | faster-whisper `base.en` | ~150 MB |
| TTS | Piper `en_US-lessac-medium` | ~63 MB |
| VAD | Silero VAD v5 ONNX | ~5 MB |

---

## Folder Structure

```
voice-agent/
│
├── models/
│   ├── llm/            # GGUF model files
│   ├── whisper/        # faster-whisper model files
│   ├── piper/          # Piper voice model + config JSON
│   └── silero/         # Silero VAD .onnx model
│
├── audio/
│   ├── __init__.py
│   ├── input_stream.py
│   ├── output_stream.py
│   └── audio_buffer.py
│
├── vad/
│   ├── __init__.py
│   └── silero_vad.py
│
├── stt/
│   ├── __init__.py
│   └── whisper_engine.py
│
├── llm/
│   ├── __init__.py
│   └── llama_engine.py
│
├── tts/
│   ├── __init__.py
│   └── piper_engine.py
│
├── core/
│   ├── __init__.py
│   ├── state_machine.py
│   ├── conversation_manager.py
│   └── realtime_orchestrator.py
│
├── config.py
├── main.py
├── requirements.txt
├── PLAN.md             # This file
└── app.py              # Original prototype (kept for reference)
```

---

## Architecture Overview

### System Flow

```
[Microphone]
     │ raw PCM chunks (512 samples @ 16kHz)
     ▼
 audio_in_q ──────────────────────────────────────────────┐
     │                                                     │ (VAD reads always)
     ▼                                                     ▼
[Silero VAD]                                     [_barge_in_watcher]
     │ SpeechStartEvent / SpeechEndEvent(audio)
     ▼
 vad_event_q
     │
     ▼
[faster-whisper STT] (ThreadPoolExecutor)
     │ transcribed text string
     ▼
 stt_result_q
     │
     ▼
[ConversationManager] → formats full prompt
     │
     ▼
[llama.cpp LLM] (ThreadPoolExecutor, async generator)
     │ streaming tokens → sentence buffer → flush on [.?!\n] or 200 chars
     ▼
 sentence_q
     │
     ▼
[Piper TTS] (ThreadPoolExecutor)
     │ PCM audio chunks
     ▼
 audio_out_q
     │
     ▼
[Speaker / sounddevice output]
```

### Pipeline Overlap (Low Latency)

```
Time →    [LLM gen s1][LLM gen s2][LLM gen s3]...
                [TTS s1]    [TTS s2]    [TTS s3]
                      [Play s1]   [Play s2]   [Play s3]
```

---

## State Machine

```
IDLE ──[speech_start]──→ LISTENING ──[speech_end]──→ TRANSCRIBING
                                                           │
                                                     THINKING (LLM)
                                                           │
                                                       SPEAKING
                                                           │
                                          [speech_start]──┘
                                                │
                                          INTERRUPTED ──→ LISTENING
```

States defined in `core/state_machine.py` as an `Enum`.

---

## Concurrency Model

### Event Loop
- Single `asyncio` event loop in the main thread
- All blocking work runs in `ThreadPoolExecutor`

### Tasks Running Concurrently

| Task | Executor | Notes |
|---|---|---|
| `_audio_capture_loop` | asyncio | sounddevice callback bridges C → asyncio queue |
| `_vad_loop` | asyncio (ONNX is fast) | Runs always, state-independent |
| `_stt_loop` | `inference_executor` (1 thread) | Blocking Whisper call |
| `_llm_loop` | `inference_executor` (1 thread) | Async generator over blocking llama.cpp |
| `_tts_loop` | `tts_executor` (1 thread) | Blocking Piper call per sentence |
| `_audio_output_loop` | asyncio | Queue → sounddevice write |
| `_barge_in_watcher` | asyncio | Active only during SPEAKING state |

### Thread Pools
- `inference_executor` — 1 worker (Whisper + LLM, never simultaneous)
- `tts_executor` — 1 worker (Piper synthesis)

### Queue Map

| Queue | Producer | Consumer | Max Size |
|---|---|---|---|
| `audio_in_q` | `_audio_capture_loop` | `_vad_loop` | 50 |
| `vad_event_q` | `_vad_loop` | `_stt_loop`, `_barge_in_watcher` | 10 |
| `stt_result_q` | `_stt_loop` | `_llm_loop` | 5 |
| `sentence_q` | `_llm_loop` | `_tts_loop` | 10 |
| `audio_out_q` | `_tts_loop` | `_audio_output_loop` | 100 |
| `control_q` | `_barge_in_watcher` | `realtime_orchestrator` | 5 |

---

## Barge-In Sequence

1. VAD fires `SpeechStartEvent` while state is `SPEAKING`
2. `_barge_in_watcher` detects it
3. Sets state → `INTERRUPTED`
4. Calls `llm_task.cancel()` + sets `llm_cancel` asyncio.Event
5. Calls `tts_task.cancel()` + sets `tts_cancel` asyncio.Event
6. Drains `audio_out_q` (calls `.get_nowait()` until empty)
7. Stops + restarts sounddevice output stream (clears hardware buffer)
8. Routes `SpeechStartEvent` (with pre-roll audio) to `AudioBuffer`
9. Sets state → `LISTENING`
10. Conversation history NOT updated with partial assistant response

---

## LLM Streaming Details

- `llama-cpp-python` called with `stream=True`
- Synchronous generator runs in `inference_executor` thread
- Each token pushed into small internal `asyncio.Queue`
- Async generator yields tokens to `_llm_loop`
- `_llm_loop` accumulates tokens into sentence buffer
- Flush triggers: `.` `?` `!` `\n` or buffer length > 200 chars
- Cancellation: `llm_cancel` asyncio.Event checked between tokens in thread

---

## TTS Streaming Details

- Piper `synthesize()` called per sentence (blocking, in `tts_executor`)
- Returns raw PCM at 22050 Hz
- Resampled to `SAMPLE_RATE` if needed via `scipy.signal.resample_poly`
- Split into 1024-sample chunks before enqueue to `audio_out_q`
- Playback of sentence N overlaps with synthesis of sentence N+1

---

## Configuration (`config.py`)

| Parameter | Value | Notes |
|---|---|---|
| `SAMPLE_RATE` | 16000 | Whisper + Silero native rate |
| `BLOCK_SIZE` | 512 | ~32ms per chunk |
| `VAD_THRESHOLD` | 0.5 | Onset probability |
| `VAD_SILENCE_CHUNKS` | 20 | ~640ms silence before SpeechEnd |
| `VAD_PRE_ROLL_CHUNKS` | 5 | ~160ms pre-roll on SpeechStart |
| `LLM_N_GPU_LAYERS` | -1 | Full Metal offload |
| `LLM_CONTEXT_SIZE` | 4096 | Conversation history window |
| `LLM_MAX_TOKENS` | 512 | Per-response cap |
| `LLM_TEMPERATURE` | 0.7 | Generation creativity |
| `SENTENCE_MAX_CHARS` | 200 | Force-flush sentence buffer |
| `TTS_SAMPLE_RATE` | 22050 | Piper native output |
| `WHISPER_MODEL` | `base.en` | Speed/accuracy balance |

---

## Latency Budget

| Stage | Expected |
|---|---|
| VAD onset detection | 32–64ms |
| VAD silence timeout | ~640ms |
| Whisper transcription (~5s audio) | 300–600ms |
| LLM first token (8B Q4_0, Metal) | 150–300ms |
| First sentence buffer fill | 300–600ms |
| Piper synthesis (one sentence) | 100–200ms |
| **Total: user stops → assistant speaks** | **~1.5–2.5 seconds** |

---

## Implementation Task List

> Tasks are ordered for sequential implementation.
> Each task builds on the previous ones.

---

### PHASE 1 — Project Scaffold

---

#### TASK 1 — `config.py`

**File:** `config.py`

**Purpose:** Central configuration. All magic numbers and paths live here.

**Contents (not functions, just module-level constants):**
- `SAMPLE_RATE = 16000`
- `BLOCK_SIZE = 512`
- `CHANNELS = 1`
- `VAD_THRESHOLD = 0.5`
- `VAD_SILENCE_CHUNKS = 20`
- `VAD_PRE_ROLL_CHUNKS = 5`
- `LLM_MODEL_PATH: Path` (resolved from `models/llm/`)
- `LLM_N_GPU_LAYERS = -1`
- `LLM_CONTEXT_SIZE = 4096`
- `LLM_MAX_TOKENS = 512`
- `LLM_TEMPERATURE = 0.7`
- `WHISPER_MODEL_SIZE = "base.en"`
- `WHISPER_MODEL_PATH: Path`
- `PIPER_MODEL_PATH: Path`
- `PIPER_CONFIG_PATH: Path`
- `SILERO_MODEL_PATH: Path`
- `SENTENCE_MAX_CHARS = 200`
- `SENTENCE_FLUSH_CHARS = ".?!\n"`
- `TTS_SAMPLE_RATE = 22050`
- `OUTPUT_CHUNK_SIZE = 1024`
- `SYSTEM_PROMPT: str`

**Dependencies:** `pathlib.Path` only

---

#### TASK 2 — `core/state_machine.py`

**File:** `core/state_machine.py`

**Purpose:** Define all assistant states and legal transitions.

**Functions / Classes:**

`class AssistantState(Enum)`
- Values: `IDLE`, `LISTENING`, `TRANSCRIBING`, `THINKING`, `SPEAKING`, `INTERRUPTED`, `STOPPING`

`class StateMachine`
- `__init__(self, initial: AssistantState)`
- `transition(self, new_state: AssistantState) -> None`
  - Input: target state
  - Output: None, raises `InvalidTransitionError` if illegal
  - Logs every transition
- `current` property → `AssistantState`
- `is_speaking() -> bool`
- `is_listening() -> bool`

`class InvalidTransitionError(Exception)` — custom exception

**Dependencies:** `enum`, `logging`

---

#### TASK 3 — Event dataclasses (inside `core/state_machine.py` or separate `core/events.py`)

**File:** `core/events.py`

**Purpose:** Typed message structs passed through queues.

**Classes (all `@dataclass`):**
- `SpeechStartEvent(pre_roll_audio: np.ndarray)`
- `SpeechEndEvent(audio: np.ndarray)`
- `TranscriptionResult(text: str)`
- `LLMToken(text: str, is_final: bool)`
- `SentenceReady(text: str)`
- `AudioChunk(pcm: np.ndarray)`
- `InterruptSignal()`
- `StopSignal()`

**Dependencies:** `dataclasses`, `numpy`

---

### PHASE 2 — Audio Layer

---

#### TASK 4 — `audio/audio_buffer.py`

**File:** `audio/audio_buffer.py`

**Purpose:** Accumulate audio chunks between SpeechStart and SpeechEnd. Manage pre-roll.

**Class:** `AudioBuffer`

- `__init__(self, pre_roll_chunks: int)`
  - Input: number of pre-roll chunks to prepend
  - Maintains internal `deque` of recent chunks for pre-roll
- `add_chunk(self, chunk: np.ndarray) -> None`
  - Input: raw audio array (512 samples)
  - Appends to pre-roll deque always; if recording, also to active buffer
- `start_recording(self) -> None`
  - Marks start; pre-roll chunks are prepended to buffer
- `stop_recording(self) -> np.ndarray`
  - Returns concatenated full audio segment
  - Resets internal buffer
- `reset(self) -> None`
  - Clears buffer without returning (used on interrupt)

**Dependencies:** `numpy`, `collections.deque`, `config`

---

#### TASK 5 — `audio/input_stream.py`

**File:** `audio/input_stream.py`

**Purpose:** Capture mic audio and push chunks to asyncio queue.

**Class:** `AudioInputStream`

- `__init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop)`
  - Stores queue and loop reference for thread-safe puts
- `start(self) -> None`
  - Opens `sounddevice.InputStream` with `SAMPLE_RATE`, `BLOCK_SIZE`, `CHANNELS`
  - Registers `_callback`
- `stop(self) -> None`
  - Closes stream cleanly
- `_callback(self, indata, frames, time, status) -> None`
  - Called by sounddevice in C thread
  - Copies `indata` to numpy array
  - Calls `loop.call_soon_threadsafe(queue.put_nowait, chunk)`

**Dependencies:** `sounddevice`, `numpy`, `asyncio`, `config`

---

#### TASK 6 — `audio/output_stream.py`

**File:** `audio/output_stream.py`

**Purpose:** Play audio chunks from asyncio queue to speaker.

**Class:** `AudioOutputStream`

- `__init__(self, queue: asyncio.Queue)`
  - Stores queue reference
  - Internal `sounddevice.OutputStream` handle
- `start(self) -> None`
  - Opens output stream at `SAMPLE_RATE`, `OUTPUT_CHUNK_SIZE`
- `stop(self) -> None`
  - Closes stream
- `flush(self) -> None`
  - Drains `audio_out_q` using `get_nowait()` until empty
  - Stops and restarts sounddevice stream to clear hardware buffer
- `is_playing(self) -> bool`
  - Returns True if output queue is non-empty or stream is active
- `async run(self) -> None`
  - Main coroutine: loops on `await queue.get()`, writes chunk to stream

**Dependencies:** `sounddevice`, `numpy`, `asyncio`, `config`

---

### PHASE 3 — VAD

---

#### TASK 7 — `vad/silero_vad.py`

**File:** `vad/silero_vad.py`

**Purpose:** Consume audio chunks, run Silero VAD, emit typed speech events.

**Class:** `SileroVAD`

- `__init__(self, model_path: Path)`
  - Loads ONNX model via `onnxruntime.InferenceSession`
  - Initializes VAD state (h, c tensors)
  - Pre-roll deque of size `VAD_PRE_ROLL_CHUNKS`
- `load(self) -> None`
  - Separate load step (called at startup before loop begins)
- `process_chunk(self, chunk: np.ndarray) -> float`
  - Input: 512-sample float32 array at 16kHz
  - Output: speech probability [0.0–1.0]
  - Runs one ONNX inference step, updates recurrent state
- `async run(self, audio_in_q: asyncio.Queue, vad_event_q: asyncio.Queue) -> None`
  - Main VAD coroutine
  - Pulls chunks from `audio_in_q`
  - Feeds `AudioBuffer.add_chunk()` always
  - Tracks `speaking` boolean
  - On onset (prob > VAD_THRESHOLD): emits `SpeechStartEvent`, calls `buffer.start_recording()`
  - On offset (silence > VAD_SILENCE_CHUNKS): emits `SpeechEndEvent(audio)`, calls `buffer.stop_recording()`

**Dependencies:** `onnxruntime`, `numpy`, `asyncio`, `config`, `audio/audio_buffer.py`, `core/events.py`

---

### PHASE 4 — Speech-to-Text

---

#### TASK 8 — `stt/whisper_engine.py`

**File:** `stt/whisper_engine.py`

**Purpose:** Transcribe audio segments to text using faster-whisper.

**Class:** `WhisperEngine`

- `__init__(self, model_size: str, model_path: Path)`
- `load(self) -> None`
  - Instantiates `faster_whisper.WhisperModel`
  - Uses `device="auto"`, `compute_type="int8"`
- `transcribe(self, audio: np.ndarray) -> str`
  - Input: float32 numpy array at 16kHz
  - Output: transcribed text string (joined segments)
  - Runs synchronously (called via executor)
- `async run(self, vad_event_q: asyncio.Queue, stt_result_q: asyncio.Queue, executor: ThreadPoolExecutor, loop: asyncio.AbstractEventLoop) -> None`
  - Waits for `SpeechEndEvent` from `vad_event_q`
  - Calls `transcribe()` via `loop.run_in_executor()`
  - Pushes `TranscriptionResult` to `stt_result_q`
  - Ignores empty transcriptions

**Dependencies:** `faster_whisper`, `numpy`, `asyncio`, `concurrent.futures`, `config`, `core/events.py`

---

### PHASE 5 — LLM

---

#### TASK 9 — `llm/llama_engine.py`

**File:** `llm/llama_engine.py`

**Purpose:** Load quantized LLM, stream tokens as async generator, support cancellation.

**Class:** `LlamaEngine`

- `__init__(self, model_path: Path)`
  - Stores path, model handle starts as None
- `load(self) -> None`
  - Instantiates `llama_cpp.Llama` with:
    - `n_gpu_layers=LLM_N_GPU_LAYERS`
    - `n_ctx=LLM_CONTEXT_SIZE`
    - `verbose=False`
- `_stream_sync(self, prompt: str, cancel_event: threading.Event) -> Generator[str, None, None]`
  - Synchronous generator (runs in executor thread)
  - Calls `self._model(prompt, stream=True, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)`
  - Yields token strings
  - Checks `cancel_event.is_set()` between each token; returns early if set
- `async stream(self, prompt: str, token_q: asyncio.Queue, cancel_event: threading.Event, executor: ThreadPoolExecutor) -> None`
  - Runs `_stream_sync` in executor
  - Each token pushed to `token_q`
  - Sentinel `LLMToken(text="", is_final=True)` pushed on completion or cancellation

**Dependencies:** `llama_cpp`, `asyncio`, `threading`, `concurrent.futures`, `config`, `core/events.py`

---

### PHASE 6 — Conversation Management

---

#### TASK 10 — `core/conversation_manager.py`

**File:** `core/conversation_manager.py`

**Purpose:** Manage conversation history and format prompts.

**Class:** `ConversationManager`

- `__init__(self, system_prompt: str, max_tokens: int)`
  - Initializes `messages: list[dict]` with system message
  - Stores `max_tokens` for context window trimming
- `add_user_message(self, text: str) -> None`
  - Appends `{"role": "user", "content": text}`
- `add_assistant_message(self, text: str) -> None`
  - Appends `{"role": "assistant", "content": text}`
- `format_prompt(self) -> str`
  - Returns full prompt string in Llama 3 chat template format
  - Uses tokens: `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`
  - Prompt ends with `<|start_header_id|>assistant<|end_header_id|>\n\n` to prime generation
- `_trim_to_context(self) -> None`
  - Removes oldest non-system messages until estimated token count is within limit
  - Token estimate: `len(content) // 4` (rough approximation)
- `reset(self) -> None`
  - Clears history, keeps system message

**Dependencies:** `config`

---

### PHASE 7 — TTS

---

#### TASK 11 — `tts/piper_engine.py`

**File:** `tts/piper_engine.py`

**Purpose:** Synthesize sentences to PCM audio chunks using Piper TTS.

**Class:** `PiperEngine`

- `__init__(self, model_path: Path, config_path: Path)`
- `load(self) -> None`
  - Instantiates `piper.PiperVoice.load(model_path, config_path)`
- `synthesize(self, text: str) -> np.ndarray`
  - Input: sentence string
  - Output: float32 PCM array at `TTS_SAMPLE_RATE`
  - Runs synchronously (called via executor)
  - Uses `io.BytesIO` + `piper.synthesize_stream_raw()` or equivalent
- `chunk_audio(self, audio: np.ndarray) -> list[np.ndarray]`
  - Input: full PCM array
  - Output: list of `OUTPUT_CHUNK_SIZE` arrays (last chunk zero-padded if needed)
- `async run(self, sentence_q: asyncio.Queue, audio_out_q: asyncio.Queue, executor: ThreadPoolExecutor, cancel_event: threading.Event) -> None`
  - Waits for `SentenceReady` from `sentence_q`
  - Calls `synthesize()` via executor
  - Checks `cancel_event` before enqueuing chunks
  - Pushes `AudioChunk` objects to `audio_out_q`

**Dependencies:** `piper`, `numpy`, `asyncio`, `threading`, `concurrent.futures`, `scipy`, `config`, `core/events.py`

---

### PHASE 8 — Orchestration

---

#### TASK 12 — `core/realtime_orchestrator.py`

**File:** `core/realtime_orchestrator.py`

**Purpose:** Central coordinator. Wires all components, manages state, implements barge-in.

**Class:** `RealtimeOrchestrator`

- `__init__(self, ...all component instances..., all queues...)`
  - Stores references to all components and queues
  - Creates `StateMachine(initial=IDLE)`
  - `self._llm_task: asyncio.Task | None = None`
  - `self._tts_task: asyncio.Task | None = None`
  - `self._llm_cancel = threading.Event()`
  - `self._tts_cancel = threading.Event()`

- `async run(self) -> None`
  - Creates all concurrent tasks via `asyncio.create_task()`
  - Awaits `asyncio.gather(*tasks)`

- `async _stt_to_llm_loop(self) -> None`
  - Waits for `TranscriptionResult` from `stt_result_q`
  - Calls `conversation_manager.add_user_message()`
  - Gets formatted prompt
  - Transitions state → `THINKING`
  - Launches `_llm_stream_loop` as cancellable task

- `async _llm_stream_loop(self) -> None`
  - Creates internal `token_q: asyncio.Queue`
  - Starts `llama_engine.stream()` in background
  - Accumulates tokens into `sentence_buffer: str`
  - On flush trigger: pushes `SentenceReady` to `sentence_q`
  - On `is_final` token: flushes remaining buffer, records full response
  - Calls `conversation_manager.add_assistant_message(full_response)`
  - Transitions state → `SPEAKING` on first sentence ready

- `async _tts_playback_loop(self) -> None`
  - Runs `piper_engine.run()` continuously
  - (Piper task is persistent; cancel_event controls per-utterance cancellation)

- `async _barge_in_watcher(self) -> None`
  - Monitors `vad_event_q` for `SpeechStartEvent`
  - If `state.is_speaking()`: calls `_handle_interrupt(event)`
  - Else: passes event through to STT pathway

- `async _handle_interrupt(self, event: SpeechStartEvent) -> None`
  - Transitions state → `INTERRUPTED`
  - Sets `_llm_cancel`, `_tts_cancel` events
  - Cancels `_llm_task` and `_tts_task` asyncio Tasks
  - Calls `audio_output.flush()`
  - Clears `_llm_cancel`, `_tts_cancel` for next turn
  - Pushes `event` back to `vad_event_q` so STT picks it up
  - Transitions state → `LISTENING`

- `async _shutdown(self) -> None`
  - Cancels all tasks
  - Stops audio streams

**Dependencies:** All other modules, `asyncio`, `threading`, `logging`

---

### PHASE 9 — Entry Point

---

#### TASK 13 — `main.py`

**File:** `main.py`

**Purpose:** Wire all components together, initialize models, start event loop.

**Functions:**

`async def main() -> None`
- Instantiates all `asyncio.Queue` objects with appropriate `maxsize`
- Creates `ThreadPoolExecutor` instances (`inference_executor`, `tts_executor`)
- Instantiates all engine classes (`SileroVAD`, `WhisperEngine`, `LlamaEngine`, `PiperEngine`)
- Calls `.load()` on each engine sequentially, logging progress
- Instantiates `AudioInputStream`, `AudioOutputStream`, `AudioBuffer`
- Instantiates `ConversationManager` with `SYSTEM_PROMPT`
- Instantiates `StateMachine`
- Instantiates `RealtimeOrchestrator` with all dependencies injected
- Starts `AudioInputStream`
- Calls `await orchestrator.run()`

`def run() -> None`
- Entry point
- Calls `asyncio.run(main())`
- Handles `KeyboardInterrupt` cleanly

**Dependencies:** All modules, `asyncio`, `logging`, `config`

---

#### TASK 14 — `requirements.txt`

**File:** `requirements.txt`

**Contents:**
```
sounddevice>=0.4.6
numpy>=1.26
faster-whisper>=1.0
llama-cpp-python>=0.2.80
piper-tts>=1.2
onnxruntime>=1.18
torch>=2.2
scipy>=1.13
soundfile>=0.12
```

**Install note for Metal llama.cpp:**
```
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## Development Roadmap

| Phase | Tasks | Goal |
|---|---|---|
| 1 | 1–3 | Scaffold: config, state machine, events |
| 2 | 4–6 | Audio pipeline: capture, buffer, playback |
| 3 | 7 | VAD: continuous speech detection |
| 4 | 8 | STT: Whisper transcription |
| 5 | 9 | LLM: streaming token generation |
| 6 | 10 | Conversation: history + prompt formatting |
| 7 | 11 | TTS: sentence synthesis + chunked playback |
| 8 | 12 | Orchestration: full pipeline wiring |
| 9 | 13–14 | Entry point + dependencies |

### Test After Each Phase
- Phase 2: Mic passthrough test (capture → play with 0 processing)
- Phase 3: VAD console log showing speech_start / speech_end events
- Phase 4: Speak → print transcription to console
- Phase 5: Type prompt → stream tokens to console
- Phase 6: Multi-turn conversation in console (no audio)
- Phase 7: Text → synthesized audio playback
- Phase 8: Full voice loop without barge-in
- Final: Barge-in stress test

---

## Notes

- `app.py` is the original prototype using Ollama + macOS `say` command. Kept for reference only.
- The new system uses only local models with no network calls.
- Llama 3 8B Instruct Q4_0 GGUF is already downloaded on this machine. Set `LLM_MODEL_PATH` in `config.py` to its location before running.
- Silero VAD ONNX model must be downloaded separately from the Silero GitHub releases.
- Piper voice model requires both `.onnx` and `.onnx.json` config files.
- **Python environment:** Use the project venv at `.venv/`. Activate with `source .venv/bin/activate` or run directly with `.venv/bin/python3`. Homebrew Python 3.14 is the base; `numpy` and `sounddevice` are already installed in the venv. Install remaining deps as each phase is implemented.
