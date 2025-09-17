from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

memory = ConversationBufferMemory(return_messages=True)

def save_to_memory(user_text: str, ai_text: str):
    memory.chat_memory.add_message(HumanMessage(content=user_text))
    memory.chat_memory.add_message(AIMessage(content=ai_text))

def get_memory():
    return memory.chat_memory.messages

def clear_memory():
    memory.chat_memory.clear()
