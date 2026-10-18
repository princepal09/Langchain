from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

prompt = PromptTemplate(
    template = 'Write a summary for the following paragraph in 10 word \n {paragraph}',
    input_variables=['paragraph']
)

parser = StrOutputParser()


loader = TextLoader('music.txt', encoding='utf-8')

docs = loader.load()
chain = prompt | model | parser

result = chain.invoke({'paragraph':docs[0].page_content})
print(result)
