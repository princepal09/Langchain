from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

# 1. Define your schema as a standard JSON Schema dictionary
review_json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review in a list",
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review",
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neu"],
            "description": "Return sentiment of the review either negative, positive or neutral",
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all pros inside a list",
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all cons inside a list",
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of reviewer",
        },
    },
    "required": ["key_themes", "summary", "sentiment"],
}

# 2. Pass the JSON schema directly into .with_structured_output()
structured_model = model.with_structured_output(review_json_schema)

# 3. Invoke the model
result = structured_model.invoke(
    """The prime suspects who were convicted and sentenced to life imprisonment in the 2022 Ankita Bhandari murder case are Pulkit Arya (the owner of the Vanantra Resort and son of a former local political leader) along with two accomplices, Saurabh Bhaskar and Ankit Gupta.
The Governor of Sikkim is not a suspect in the murder, but has recently become the center of a major political controversy."""
)

print(result)