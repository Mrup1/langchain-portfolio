from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader

loader = DirectoryLoader(
    path = 'resume',
    glob = '*.pdf',
    loader_cls= PyPDFLoader
)

docs = loader.load()

splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separator=''
)

result = splitter.split_documents(docs)

print((result[3]))
