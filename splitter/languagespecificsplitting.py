from langchain.text_splitter import RecursiveCharacterTextSplitter , Language

text = """
from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

loader = DirectoryLoader(
    path = 'resume',
    glob = '*.pdf',
    loader_cls= PyPDFLoader
)

doc = loader.load()

print(len(doc))

Prompt = PromptTemplate(
    template='tell me about sudais qualifications \n {qualification}',
    input_variables=['qualification']
)
parser = StrOutputParser()

chain = Prompt | model | parser 

result = chain.invoke({'qualification':doc})
print(result)


"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,#can use this for markdown also just write MARKDOWN in place of the python
    chunk_size = 100,
    chunk_overlap = 0
)

result = splitter.split_text(text)

print((result[10]))