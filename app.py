import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import base64
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from utils.tts import stream_speak
from memory.memory import save_to_memory, get_memory, clear_memory
from dotenv import load_dotenv
load_dotenv()

# LLM
chat = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# Page setup
st.set_page_config(page_title="🤖 Multimodal Chatbot", layout="wide")
st.title("💬 Multimodal Chatbot")

# Session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "extra_mode" not in st.session_state:
    st.session_state["extra_mode"] = False
if "chat_input" not in st.session_state:
    st.session_state["chat_input"] = ""


# --- Sidebar ---
st.sidebar.header("⚙️ Controls")
if st.sidebar.button("🆕 New Chat"):
    clear_memory()
    st.session_state["messages"] = []
    st.sidebar.success("Started new chat!")

if st.sidebar.button("🗑️ Clear Chat"):
    clear_memory()
    st.session_state["messages"] = []
    st.sidebar.success("Chat cleared!")


# --- Chat History ---
for role, msg in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(msg)


# --- Input Handling ---
def submit_text():
    user_text = st.session_state.chat_input.strip()
    if not user_text:
        return

    st.session_state["messages"].append(("user", user_text))
    st.chat_message("user").markdown(user_text)

    with st.chat_message("assistant"):
        resp_container = st.empty()
        full_reply = ""

        for chunk in chat.stream(get_memory() + [HumanMessage(content=user_text)]):
            if chunk.content:
                full_reply += chunk.content
                resp_container.markdown(full_reply)
                stream_speak(chunk.content)

        save_to_memory(user_text, full_reply)
        st.session_state["messages"].append(("assistant", full_reply))

    st.session_state.chat_input = ""  # clear after send


def handle_file_upload(uploaded_file):
    if uploaded_file:
        file_b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")

        st.session_state["messages"].append(("user", f"📎 File uploaded: {uploaded_file.name}"))
        st.chat_message("user").markdown(f"📎 File uploaded: {uploaded_file.name}")

        with st.chat_message("assistant"):
            resp_container = st.empty()
            full_reply = ""

            message = HumanMessage(content=[
                {"type": "text", "text": f"Describe this file: {uploaded_file.name}"},
                {"type": "image_url", "image_url": f"data:application/octet-stream;base64,{file_b64}"}
            ])
            for chunk in chat.stream(get_memory() + [message]):
                if chunk.content:
                    full_reply += chunk.content
                    resp_container.markdown(full_reply)

            save_to_memory(f"File: {uploaded_file.name}", full_reply)
            st.session_state["messages"].append(("assistant", full_reply))


def handle_audio(audio_data):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_data["filename"]) as source:
        audio = recognizer.record(source)
        user_text = recognizer.recognize_google(audio)

    st.session_state["messages"].append(("user", f"🎙️ {user_text}"))
    st.chat_message("user").markdown(f"🎙️ {user_text}")

    with st.chat_message("assistant"):
        resp_container = st.empty()
        full_reply = ""
        for chunk in chat.stream(get_memory() + [HumanMessage(content=user_text)]):
            if chunk.content:
                full_reply += chunk.content
                resp_container.markdown(full_reply)
                stream_speak(chunk.content)

        save_to_memory(user_text, full_reply)
        st.session_state["messages"].append(("assistant", full_reply))


# --- Custom CSS ---
st.markdown(
    """
    <style>
    .chat-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 10px;
        border-top: 1px solid #ddd;
        z-index: 999;
    }
    .chat-input-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .plus-btn {
        border-radius: 50%;
        height: 40px;
        width: 40px;
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        cursor: pointer;
    }
    .dropdown {
        position: absolute;
        bottom: 60px;
        left: 10px;
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Input Bar (Fixed at Bottom) ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 9])
with col1:
    if st.button("➕", key="plus_btn"):
        st.session_state.extra_mode = not st.session_state.extra_mode

with col2:
    st.text_input(
        "Chat Input",  # hidden but required
        key="chat_input",
        on_change=submit_text,
        placeholder="Type message and press Enter...",
        label_visibility="collapsed"
    )

st.markdown('</div>', unsafe_allow_html=True)


# --- Dropdown Menu under + button ---
if st.session_state.extra_mode:
    st.markdown('<div class="dropdown">', unsafe_allow_html=True)
    choice = st.radio("Select input:", ["🎙️ Audio", "📂 File"], label_visibility="collapsed")

    if choice == "🎙️ Audio":
        audio_data = mic_recorder(start_prompt="🎙️ Record", stop_prompt="🛑 Stop", just_once=True)
        if audio_data:
            handle_audio(audio_data)

    elif choice == "📂 File":
        uploaded_file = st.file_uploader("Upload file", type=["png", "jpg", "jpeg", "pdf", "txt"])
        if uploaded_file:
            handle_file_upload(uploaded_file)

    st.markdown('</div>', unsafe_allow_html=True)
