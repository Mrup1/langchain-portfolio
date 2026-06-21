#RunnableLambda is a runnable primitive that allows you to apply custom Python functions within an Al pipeline.
#It acts as a middleware between different Al components, enabling preprocessing, transformation, API calls, filtering, and post-processing in a LangChain workflow
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnableLambda , RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def word_count(x):
    return len(x.split())

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

prompt1 = PromptTemplate(
    template= 'tell me a joke about the topic {topic}',
    input_variables= ['topic']
)

parser = StrOutputParser()

joke_chain = RunnableSequence (prompt1 | model | parser)

chain2 = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_chain , chain2)

result = final_chain.invoke({'topic':'airpods'})

print(result)