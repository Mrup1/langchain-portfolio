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

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'tell me a joke on the {topic}',
    input_variables  = ['topic']
)

pormpt2 = PromptTemplate(
    template = 'explain the following text {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1 | model | parser | pormpt2 | model | parser)

result = chain.invoke({'topic':'computer'})

print(result)

#runablesequence helps us to make a sequential chain 