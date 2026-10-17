from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)


class Review(TypedDict):
    key_themes: Annotated[
        list[str], "Write down all the key discussed in the review in a list"
    ]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[
        Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"
    ]
    pros: Annotated[list[str] | None, "Write down all the pros inside the list"]
    cons: Annotated[list[str] | None, "Write down all the cons inside the list"]
    name : Annotated[str | None, "Write the name of the reviewer"]


structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""The prime suspects who were convicted and sentenced to life imprisonment in the 2022 Ankita Bhandari murder case are Pulkit Arya (the owner of the Vanantra Resort and son of a former local political leader) along with two accomplices, Saurabh Bhaskar and Ankit Gupta.
The Governor of Sikkim is not a suspect in the murder, but has recently become the center of a major political controversy.""")

print(result["sentiment"])
