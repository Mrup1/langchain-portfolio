from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv() # Load your API keys from .env file

# Create raw document data as dictionaries
doc1 = {
    "content": "Artificial intelligence (AI) is transforming industries worldwide, from healthcare to finance. Its applications range from predictive analytics to natural language processing, enhancing efficiency and decision-making.",
    "metadata": {"source": "AI_Insights", "page": 1, "topic": "Technology"}
}

doc2 = {
    "content": "Renewable energy sources like solar and wind power are crucial for combating climate change. Governments and private sectors are investing heavily in these technologies to reduce reliance on fossil fuels.",
    "metadata": {"source": "Energy_Future", "page": 15, "topic": "Environment"}
}

doc3 = {
    "content": "The human brain is an incredibly complex organ, responsible for thoughts, emotions, and memories. Neuroscientists are continuously uncovering its mysteries, leading to breakthroughs in understanding neurological disorders.",
    "metadata": {"source": "Brain_Science_Daily", "page": 3, "topic": "Biology"}
}

doc4 = {
    "content": "Space exploration continues to push the boundaries of human knowledge. Recent missions to Mars and the discovery of exoplanets offer tantalizing glimpses into the universe beyond our solar system.",
    "metadata": {"source": "Cosmic_Voyages", "page": 8, "topic": "Astronomy"}
}

doc5 = {
    "content": "The art of pottery has a rich history, dating back thousands of years across various cultures. From functional vessels to elaborate sculptures, ceramics reflect artistic expression and cultural heritage.",
    "metadata": {"source": "Art_History_Review", "page": 22, "topic": "Culture"}
}

doc6 = {
    "content": "Financial markets are influenced by a multitude of factors, including economic indicators, geopolitical events, and investor sentiment. Understanding these dynamics is key to successful investment strategies.",
    "metadata": {"source": "Market_Watch", "page": 5, "topic": "Economy"}
}

doc7 = {
    "content": "Classical music, with its intricate compositions and emotional depth, has captivated audiences for centuries. Composers like Beethoven and Mozart left an indelible mark on the world's musical landscape.",
    "metadata": {"source": "Music_Legacy", "page": 10, "topic": "Arts"}
}

# Combine all initial document data dictionaries
all_initial_docs_data = [doc1, doc2, doc3, doc4, doc5, doc6, doc7]

# Initialize Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Chroma vector store
# It will create 'my_chroma_db' directory if it doesn't exist
print("Initializing ChromaDB and preparing documents...")
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='general_knowledge' # Renamed for clarity as it's not IPL specific
)

# Convert your dictionary documents into LangChain Document objects
# This step creates the `page_content` and `metadata` attributes that Chroma expects.
langchain_documents_to_add = [
    Document(page_content=doc_data["content"], metadata=doc_data["metadata"])
    for doc_data in all_initial_docs_data
]

# Add the converted LangChain Document objects to the vector store
print("Adding documents to ChromaDB...")
vector_store.add_documents(langchain_documents_to_add)
print("Documents added to ChromaDB. ✅")

# View documents stored in ChromaDB
print("\n--- Viewing all documents in the vector store ---")
retrieved_data = vector_store.get(
    include=['embeddings', 'documents', 'metadatas']
)

print(f"Total documents retrieved: {len(retrieved_data['documents'])}")
for i in range(len(retrieved_data['documents'])):
    print(f"\nDocument {i+1}:")
    print(f"  Content: {retrieved_data['documents'][i]}")
    print(f"  Metadata: {retrieved_data['metadatas'][i]}")
    # print(f"  Embedding (first 10 elements): {retrieved_data['embeddings'][i][:10]}...") # Uncomment to see embeddings

print("\nCode execution complete. Check 'my_chroma_db' directory for persisted data.")

# Query 1: Similarity search for financial information
query_financial = 'tell me about financial markets?'
print(f"\n--- Searching for: '{query_financial}' (top 2 results) ---")
financial_results = vector_store.similarity_search(
    query=query_financial,
    k=2
)
for i, doc in enumerate(financial_results):
    print(f"\nResult {i+1} (Similarity Search):")
    print(f"  Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")


# Query 2: Similarity search with metadata filtering
# Note: The 'team' metadata was from the previous IPL example.
# For the 'general_knowledge' collection, we should use existing metadata keys like 'topic'.
# Let's search for documents related to 'Technology' with a score.
query_tech_filtered = "What are the latest advancements in AI?"
print(f"\n--- Searching for: '{query_tech_filtered}' with filter (topic: Technology) and score ---")
filtered_results_with_score = vector_store.similarity_search_with_score(
    query=query_tech_filtered,
    k=2, # Get top 2 results
    filter={"topic": "Technology"} # Filter by the 'topic' metadata
)

if filtered_results_with_score:
    for i, (doc, score) in enumerate(filtered_results_with_score):
        print(f"\nResult {i+1} (Filtered Search with Score):")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")
        print(f"  Similarity Score: {score}")
else:
    print("\nNo documents found matching the filtered query.")