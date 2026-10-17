from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(model="qwen2.5:3b", temperature=0)

messages = [
    SystemMessage(content = 'You are a helpful assitant'),
    HumanMessage(content = 'Tell me about langhchain'),
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)