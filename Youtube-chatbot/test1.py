from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

df = pd.read_csv("transcripts.csv")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2,
    chunk_overlap=1,
    separators="\n\n"
)
chunks = splitter.create_documents(df["Structured_Transcript"].tolist())
print(os.getenv("GOOGLE_API_KEY"))  # Ensure the API key is loaded correctly
# Try specifying the API key explicitly
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
result = retriever.invoke('Who is Fredric Schiffer?')

# for doc in result:
#     print(doc.page_content)