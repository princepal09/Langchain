from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(model = "qwen2.5:3b", temperature = 0)

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"],
)

parser = StrOutputParser()

url = "https://www.flipkart.com/apple-macbook-air-m5-2026-m5-16-gb-512-gb-ssd-tahoe-mdh74hn-a/p/itm7780abdd2be7d?pid=COMHZQX4GXGGJHDY&lid=LSTCOMHZQX4GXGGJHDYEWXLB8&marketplace=FLIPKART&store=6bo%2Fb5g&srno=b_1_1&otracker=browse&fm=organic&iid=7b44765d-699a-4df5-bfc2-2ccb9daeb27a.COMHZQX4GXGGJHDY.SEARCH&ppt=None&ppn=None&ssid=q8ghvamy800000001792303738467&ov_redirect=true"

loader = WebBaseLoader(url)


docs = loader.load()

chain = prompt | model | parser

result = chain.invoke(
    {
        "question": "What is the product that we are talking about?",
        "text": docs[0].page_content,
    }
)
print(result)
