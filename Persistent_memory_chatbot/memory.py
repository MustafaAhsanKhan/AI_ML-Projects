# In-memory store:
# key   -> session_id (for example: "user1")
# value -> list of message dictionaries for that session
sessions = {}

def get_history(session_id):
    # Return all messages for this session.
    # If session does not exist yet, return an empty list.
    return sessions.get(session_id, [])

def append_message(session_id, role, content):
    # Create session entry the first time we see this session_id.
    if session_id not in sessions:
        sessions[session_id] = []

    # Add one message as a dictionary with role and text content.
    sessions[session_id].append({
        "role": role,
        "content": content
    })