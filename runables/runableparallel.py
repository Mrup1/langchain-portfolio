#RunnableParallel is a runnable primitive(that connents two runable tasks) that allows multiple runnables to execute in parallel.
#Each runnable receives the same input and processes it independently, producing a dictionary of outputs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

prompt1 = PromptTemplate(
    template= 'tell a good thing about the topic {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= 'tell a bad thing about the topic {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'good' : RunnableSequence(prompt1 | model | parser),
    'bad'  : RunnableSequence(prompt2 | model | parser)
})

result = chain.invoke({'topic': 'AI'})

print(result['good'])
print(result['bad']) 