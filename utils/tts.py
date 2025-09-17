import pyttsx3

engine = pyttsx3.init()

def stream_speak(text: str):
    """Speak text as it streams in real-time."""
    engine.say(text)
    engine.runAndWait()
