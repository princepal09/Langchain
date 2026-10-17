from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


documents = [
    "Python is a popular programming language used for web development, data science, automation, and artificial intelligence.",
    "The Earth revolves around the Sun once every 365.25 days. This movement is responsible for the length of a year.",
    "Coffee is one of the world's most popular beverages. It contains caffeine, which can help increase alertness and concentration.",
    "Artificial intelligence allows computers to perform tasks that normally require human intelligence, such as understanding language and recognizing images.",
    "The Himalayas are a major mountain range in Asia and include Mount Everest, the highest mountain above sea level.",
    "Basketball is a team sport in which two teams try to score points by shooting a ball through a hoop.",
    "Reading books regularly can improve vocabulary, comprehension, and general knowledge.",
    "The Internet connects billions of devices around the world and enables people to communicate and share information.",
]

query = "Tell me about himalayas"

query_embeddings = embedding.embed_query(query)
doc_embeddings = embedding.embed_documents(documents)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

index, score = max(list(enumerate(scores)), key=lambda x: x[1])

print(query)
print(documents[index])
print("Similarity score is:", score)
