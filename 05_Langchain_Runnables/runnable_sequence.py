from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke {text}", input_variables=["text"]
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

result = chain.invoke({"topic": "Music"})
print(result)
