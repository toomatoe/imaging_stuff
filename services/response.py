from models import Context

def generate_response(context: Context) -> str:
    mood = context.emotion
    text = context.transcript

    if mood == "happy":
        return "You sound cheerful! What's going well?"
    elif mood == "sad":
        return "I'm here for you. Want to talk about it?"
    elif mood == "angry":
        return "I hear some frustration. Want me to slow down?"
    else:
        return f"I hear you. You said: '{text}'. Let's explore that."