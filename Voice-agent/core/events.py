from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpeechStartEvent:
    """VAD detected speech onset. Carries pre-roll audio captured before detection."""
    pre_roll_audio: np.ndarray = field(repr=False)


@dataclass
class SpeechEndEvent:
    """VAD detected end of utterance. Carries the full audio segment for STT."""
    audio: np.ndarray = field(repr=False)


@dataclass
class TranscriptionResult:
    """Whisper produced a transcription from an audio segment."""
    text: str


@dataclass
class LLMToken:
    """A single token streamed from the LLM."""
    text: str
    is_final: bool = False      # True on the last token or after cancellation


@dataclass
class SentenceReady:
    """A complete sentence flushed from the LLM token buffer, ready for TTS."""
    text: str


@dataclass
class AudioChunk:
    """A fixed-size block of PCM audio ready for playback."""
    pcm: np.ndarray = field(repr=False)


@dataclass
class InterruptSignal:
    """Barge-in detected during SPEAKING — cancel LLM + TTS and resume listening."""
    pass


@dataclass
class StopSignal:
    """Clean shutdown requested."""
    pass
