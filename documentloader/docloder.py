from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('football.txt')

doc = loader.load()

print(type(doc))

print(type(doc[0]))

print(len(doc))

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

parser = StrOutputParser()

Prompt = PromptTemplate(
    template='write me the summary of the topic \n{topic}',
    input_variables=['topic']
)

chain = RunnableSequence(Prompt , model , parser)

result = chain.invoke({'topic':doc[0].page_content})

print(result)



