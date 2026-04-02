import re
from collections import Counter

# In-memory store:
# key   -> session_id (for example: "user1")
# value -> list of message dictionaries for that session
sessions = {}

# Per-session semantic memory state.
session_summaries = {}
session_last_summarized_index = {}
session_semantic_state = {}

# Tuning knobs for a small, in-memory chatbot.
SUMMARY_EVERY_MESSAGES = 8
RECENT_MESSAGE_WINDOW = 6
EARLY_SUMMARY_CHAR_THRESHOLD = 1400

STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "your", "have", "what", "when", "where",
    "which", "would", "could", "there", "their", "about", "into", "while", "should", "because",
    "user", "assistant", "just", "also", "then", "than", "they", "them", "been", "were", "will",
    "please", "need", "want", "like", "does", "did", "you", "are", "for", "not", "our", "out",
    "can", "how", "who", "why", "his", "her", "she", "him", "its", "had", "has", "was", "all",
    "any", "but", "too", "very", "more", "most", "some", "such", "use", "using", "used", "today",
}

IDENTITY_MARKERS = (
    "my name is",
    "i am",
    "i'm",
    "call me",
    "i prefer",
    "my favorite",
    "i like",
    "i love",
    "i work as",
    "i live in",
)

DECISION_MARKERS = (
    "we decided",
    "we will",
    "i will",
    "i'll",
    "let's",
    "lets",
    "we should",
    "go with",
    "choose",
    "chosen",
    "final",
    "conclusion",
    "plan",
)


def _ensure_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = []
    if session_id not in session_summaries:
        session_summaries[session_id] = ""
    if session_id not in session_last_summarized_index:
        session_last_summarized_index[session_id] = 0
    if session_id not in session_semantic_state:
        session_semantic_state[session_id] = {
            "identity": [],
            "decisions": [],
            "topic_counts": Counter(),
        }


def get_history(session_id):
    # Return all messages for this session.
    # If session does not exist yet, return an empty list.
    return sessions.get(session_id, [])


def append_message(session_id, role, content):
    # Create session entry the first time we see this session_id.
    _ensure_session(session_id)

    # Add one message as a dictionary with role and text content.
    sessions[session_id].append({
        "role": role,
        "content": content
    })


def _split_sentences(text):
    return [s.strip() for s in re.split(r"[.!?]\s+", text.strip()) if s.strip()]


def _extract_identity_facts(messages):
    facts = []
    seen = set()

    for message in messages:
        if message.get("role") != "user":
            continue

        content = message.get("content", "").strip()
        lower_content = content.lower()

        name_match = re.search(r"\bmy name is\s+([a-zA-Z][a-zA-Z\s\-']{0,40})", content, flags=re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip().rstrip(".,!?")
            name_fact = f"Name: {name}"
            key = name_fact.lower()
            if key not in seen:
                facts.append(name_fact)
                seen.add(key)

        if any(marker in lower_content for marker in IDENTITY_MARKERS):
            first_sentence = _split_sentences(content)[0] if content else ""
            if first_sentence:
                fact = first_sentence[:180]
                key = fact.lower()
                if key not in seen:
                    facts.append(fact)
                    seen.add(key)

    return facts


def _extract_decisions(messages):
    decisions = []
    seen = set()

    for message in messages:
        content = message.get("content", "").strip()
        if not content:
            continue

        for sentence in _split_sentences(content):
            lower_sentence = sentence.lower()
            if any(marker in lower_sentence for marker in DECISION_MARKERS):
                decision = sentence[:180]
                key = decision.lower()
                if key not in seen:
                    decisions.append(decision)
                    seen.add(key)

    return decisions


def _extract_topic_counts(messages):
    counter = Counter()

    for message in messages:
        content = message.get("content", "")
        words = re.findall(r"[a-zA-Z]{3,}", content.lower())
        for word in words:
            if word in STOPWORDS:
                continue
            counter[word] += 1

    return counter


def _render_summary(session_id):
    state = session_semantic_state[session_id]

    identity_items = state["identity"][:5]
    decision_items = state["decisions"][:5]
    topics = [topic for topic, _ in state["topic_counts"].most_common(6)]

    identity_text = "; ".join(identity_items) if identity_items else "No critical identity facts captured yet."
    topics_text = ", ".join(topics) if topics else "No major recurring topics yet."
    decisions_text = "; ".join(decision_items) if decision_items else "No explicit decisions or conclusions yet."

    return (
        "User Identity: " + identity_text + "\n"
        "Key Topics: " + topics_text + "\n"
        "Decisions/Conclusions: " + decisions_text
    )


def should_summarize(session_id):
    _ensure_session(session_id)

    history = sessions[session_id]
    summarize_upto = max(0, len(history) - RECENT_MESSAGE_WINDOW)
    start = session_last_summarized_index[session_id]

    if summarize_upto <= start:
        return False

    candidate_messages = history[start:summarize_upto]
    candidate_chars = sum(len(msg.get("content", "")) for msg in candidate_messages)

    if len(candidate_messages) >= SUMMARY_EVERY_MESSAGES:
        return True

    return len(candidate_messages) >= 4 and candidate_chars >= EARLY_SUMMARY_CHAR_THRESHOLD


def update_summary(session_id):
    _ensure_session(session_id)

    if not should_summarize(session_id):
        return session_summaries[session_id]

    history = sessions[session_id]
    start = session_last_summarized_index[session_id]
    summarize_upto = max(0, len(history) - RECENT_MESSAGE_WINDOW)
    candidate_messages = history[start:summarize_upto]

    if not candidate_messages:
        return session_summaries[session_id]

    state = session_semantic_state[session_id]

    for fact in _extract_identity_facts(candidate_messages):
        if fact not in state["identity"]:
            state["identity"].append(fact)

    for decision in _extract_decisions(candidate_messages):
        if decision not in state["decisions"]:
            state["decisions"].append(decision)

    state["topic_counts"].update(_extract_topic_counts(candidate_messages))

    session_summaries[session_id] = _render_summary(session_id)
    session_last_summarized_index[session_id] = summarize_upto

    return session_summaries[session_id]


def get_summary(session_id):
    _ensure_session(session_id)
    return session_summaries[session_id]


def build_model_context(session_id):
    _ensure_session(session_id)

    summary = session_summaries[session_id]
    recent_messages = list(sessions[session_id][-RECENT_MESSAGE_WINDOW:])

    if not summary:
        return recent_messages

    summary_message = {
        "role": "system",
        "content": (
            "Summary of earlier conversation context. Use this as background memory while replying.\n"
            + summary
        ),
    }

    return [summary_message] + recent_messages