from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """
The Amazon rainforest is a vast natural wonder, home to an incredible diversity of plant and animal species. It plays a crucial role in regulating the Earth's climate by absorbing carbon dioxide and producing oxygen. Deforestation and climate change pose significant threats to this irreplaceable ecosystem.

Quantum computing is an emerging technology that leverages the principles of quantum mechanics to solve complex problems far beyond the capabilities of classical computers. Instead of bits, it uses qubits, which can exist in multiple states simultaneously, enabling exponential computational power. Research in this field is rapidly advancing, promising breakthroughs in medicine, materials science, and cryptography.

The invention of the printing press by Johannes Gutenberg in the 15th century revolutionized the spread of knowledge. It made books more accessible and affordable, leading to increased literacy and the rapid dissemination of ideas during the Renaissance and beyond. This invention fundamentally changed the course of human history.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)