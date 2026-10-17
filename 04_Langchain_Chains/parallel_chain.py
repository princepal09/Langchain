from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

model2 = ChatOllama(model="qwen2.5:3b", temperature=0)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions answers from the following text \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)


parser = StrOutputParser()


parallel_chain = RunnableParallel(
    {"notes": prompt1 | model1 | parser, "quiz": prompt2 | model2 | parser}
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
TypeScript has fundamentally transformed the landscape of modern web development by bridging the gap between dynamic scripting and enterprise-grade reliability, serving as a strongly typed superset of JavaScript that compiles down to plain, highly compatible code executable in any browser or server-side runtime like Node.js. By introducing static types, interfaces, generics, and advanced type-inference mechanisms directly into the developer experience, it empowers full-stack engineers to catch subtle bugs, type mismatches, and structural errors right inside their code editors long before the application ever reaches a staging or production environment. This proactive error detection drastically reduces runtime exceptions, enhances refactoring confidence across massive codebases, and provides crystal-clear documentation that streamlines collaboration within development teams. Furthermore, TypeScript seamlessly integrates with modern frontend frameworks like React, UI component libraries, and robust backend ecosystems utilizing Express, Prisma, and PostgreSQL, allowing developers to maintain end-to-end type safety from database schemas all the way to client-side user interfaces without sacrificing the flexibility and speed that make JavaScript the undisputed language of the web."""

result = chain.invoke({"text": text})

print(result)