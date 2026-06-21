from langchain_community.document_loaders import WebBaseLoader
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
url = 'https://www.apple.com/shop/buy-mac/macbook-pro/14-inch-m4-pro'
loader = WebBaseLoader(url)

doc = loader.load()

print(len(doc))

Prompt = PromptTemplate(
    template='tell me about the question {question}\n from the following text {text}',
    input_variables=['question' , 'text']
)
parser = StrOutputParser()

chain = Prompt | model | parser 

result = chain.invoke({'question':'what is the price and the colours of the mac m4?', 'text' : doc})
print(result)

