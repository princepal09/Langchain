from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of feedback"
    )


parser = StrOutputParser()
structure_model = model.with_structured_output(Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback}",
    input_variables=["feedback"],
)


classifier_chain = prompt1 | structure_model

prompt2 = PromptTemplate(
    template="Write an appropriate respone to this positive feedback \n{feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate respone to this negative feedback \n{feedback}",
    input_variables=["feedback"],
)

positive_chain = prompt2 | model | parser
negative_chain = prompt3 | model | parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", positive_chain),
    (lambda x: x.sentiment == "negative", negative_chain),
    RunnableLambda(lambda x: "Could not find sentiment"),
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "This is a Wonderful phone"})
print(result)
