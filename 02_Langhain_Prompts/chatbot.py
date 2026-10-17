from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(model="qwen2.5:3b", temperature=0)

chat_history = [SystemMessage(content="You are a helful AI Assitant")]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI: ", result.content)

print(chat_history)
