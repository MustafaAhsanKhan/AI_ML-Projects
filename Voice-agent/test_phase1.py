import sys
sys.path.insert(0, "/Users/mustafa/Desktop/AI_ML-Projects/Voice-agent")

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

import numpy as np

# --- config ---
import config
assert config.SAMPLE_RATE == 16000
assert config.BLOCK_SIZE == 512
assert config.CHANNELS == 1
assert config.VAD_THRESHOLD == 0.5
assert config.LLM_N_GPU_LAYERS == -1
assert config.SENTENCE_MAX_CHARS == 200
print("config OK")

# --- state machine ---
from core.state_machine import StateMachine, AssistantState, InvalidTransitionError

# happy path
sm = StateMachine()
sm.transition(AssistantState.LISTENING)
sm.transition(AssistantState.TRANSCRIBING)
sm.transition(AssistantState.THINKING)
sm.transition(AssistantState.SPEAKING)
sm.transition(AssistantState.IDLE)
assert sm.current == AssistantState.IDLE
print("state machine happy path OK")

# barge-in path
sm2 = StateMachine()
sm2.transition(AssistantState.LISTENING)
sm2.transition(AssistantState.TRANSCRIBING)
sm2.transition(AssistantState.THINKING)
sm2.transition(AssistantState.SPEAKING)
sm2.transition(AssistantState.INTERRUPTED)
sm2.transition(AssistantState.LISTENING)
assert sm2.is_listening()
print("state machine barge-in path OK")

# empty transcription path
sm3 = StateMachine()
sm3.transition(AssistantState.LISTENING)
sm3.transition(AssistantState.TRANSCRIBING)
sm3.transition(AssistantState.IDLE)
assert sm3.is_idle()
print("state machine empty-transcription path OK")

# invalid transition
sm4 = StateMachine()
try:
    sm4.transition(AssistantState.SPEAKING)
    raise AssertionError("should have raised")
except InvalidTransitionError as e:
    print("invalid transition correctly blocked OK:", e)

# helper predicates
sm5 = StateMachine()
sm5.transition(AssistantState.LISTENING)
assert sm5.is_listening()
assert not sm5.is_speaking()
assert not sm5.is_idle()
print("state machine predicates OK")

# --- events ---
from core.events import (
    SpeechStartEvent, SpeechEndEvent, TranscriptionResult,
    LLMToken, SentenceReady, AudioChunk, InterruptSignal, StopSignal,
)

dummy = np.zeros(512, dtype=np.float32)

e1 = SpeechStartEvent(pre_roll_audio=dummy)
e2 = SpeechEndEvent(audio=dummy)
e3 = TranscriptionResult(text="hello world")
e4 = LLMToken(text=" hello", is_final=False)
e5 = LLMToken(text="", is_final=True)
e6 = SentenceReady(text="Hello there.")
e7 = AudioChunk(pcm=dummy)
e8 = InterruptSignal()
e9 = StopSignal()

assert e3.text == "hello world"
assert e4.is_final is False
assert e5.is_final is True
assert e6.text == "Hello there."
assert isinstance(e8, InterruptSignal)
print("events OK")

print()
print("=== All Phase 1 tests passed ===")
