import speech_recognition as sr

def listen() -> str:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak...")
        audio = recognizer.listen(source)
    return recognizer.recognize_google(audio)
