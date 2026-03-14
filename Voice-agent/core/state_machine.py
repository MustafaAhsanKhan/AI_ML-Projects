import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()
    STOPPING = auto()


# Legal transitions: state -> set of states it may move to
_VALID_TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.IDLE: {
        AssistantState.LISTENING,
        AssistantState.STOPPING,
    },
    AssistantState.LISTENING: {
        AssistantState.TRANSCRIBING,
        AssistantState.IDLE,           # empty audio / below threshold
        AssistantState.STOPPING,
    },
    AssistantState.TRANSCRIBING: {
        AssistantState.THINKING,
        AssistantState.IDLE,           # empty transcription
        AssistantState.STOPPING,
    },
    AssistantState.THINKING: {
        AssistantState.SPEAKING,
        AssistantState.IDLE,           # LLM error / empty response
        AssistantState.INTERRUPTED,    # barge-in before first token
        AssistantState.STOPPING,
    },
    AssistantState.SPEAKING: {
        AssistantState.IDLE,           # response complete
        AssistantState.INTERRUPTED,    # barge-in mid-speech
        AssistantState.STOPPING,
    },
    AssistantState.INTERRUPTED: {
        AssistantState.LISTENING,      # always continue to next utterance
        AssistantState.STOPPING,
    },
    AssistantState.STOPPING: set(),
}


class InvalidTransitionError(Exception):
    pass


class StateMachine:
    def __init__(self, initial: AssistantState = AssistantState.IDLE) -> None:
        self._state = initial
        logger.info("StateMachine initialised — state: %s", self._state.name)

    @property
    def current(self) -> AssistantState:
        return self._state

    def transition(self, new_state: AssistantState) -> None:
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.name} to {new_state.name}"
            )
        logger.debug("State: %s → %s", self._state.name, new_state.name)
        self._state = new_state

    def is_speaking(self) -> bool:
        return self._state == AssistantState.SPEAKING

    def is_listening(self) -> bool:
        return self._state == AssistantState.LISTENING

    def is_idle(self) -> bool:
        return self._state == AssistantState.IDLE

    def is_stopping(self) -> bool:
        return self._state == AssistantState.STOPPING
