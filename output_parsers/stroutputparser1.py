from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

template1 = PromptTemplate(
    template='write me a detail summary of the {topic}',
    input_variables=['topic']
)
template2 = PromptTemplate(
    template='write me a 5 line summary of the \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic':'black hole'})

print(result)