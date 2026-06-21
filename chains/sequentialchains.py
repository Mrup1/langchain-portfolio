from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

parser = StrOutputParser()

template = PromptTemplate(
    template='tell me 10 point about the {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='now tell me the 5 best point about the pervious 10 points \n {text}',
    input_variables=['text']
)

chain = template | model | parser | template2 | model | parser

chain.get_graph().print_ascii() # this is to visualize the chain

result = chain.invoke ({'topic': 'science and technology'})

print(result) 

#chain.get_graph().print_ascii()