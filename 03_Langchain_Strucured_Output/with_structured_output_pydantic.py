from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal("pos", "neg") = Field( # type: ignore
        description="Return sentiment of the review either negative, positive or neutral"
    ) 
    pros: str | None = Field(
        default=None, description="Write down all pros inside a list"
    )
    cons: str | None = Field(
        default=None, description="Write down all cons inside a list"
    )
    name: str | None = Field(default=None, description="Write the name of reviewer")


structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""The prime suspects who were convicted and sentenced to life imprisonment in the 2022 Ankita Bhandari murder case are Pulkit Arya (the owner of the Vanantra Resort and son of a former local political leader) along with two accomplices, Saurabh Bhaskar and Ankit Gupta.
The Governor of Sikkim is not a suspect in the murder, but has recently become the center of a major political controversy.""")


