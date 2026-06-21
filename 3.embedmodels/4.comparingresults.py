from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#load_dotenv()

model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Japan is Tokyo."
]
query = "What is the capital of France?"
query_embedding = model.embed_query(query)
documents_embedding= model.embed_documents(documents)

scores = cosine_similarity([query_embedding], documents_embedding)[0]

index , score =sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Query: {query}")
print(f"Most similar document: {documents[index]}")
print(f"Cosine similarity score: {score:.4f}")
      

