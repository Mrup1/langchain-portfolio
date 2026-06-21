#runable pass through gives the same output that is given to it in input 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

prompt = PromptTemplate(
    template= 'tell me a joke about toopoic {topic}',
    input_variables= ['topic']
)
prompt2 = PromptTemplate(
    template='write me the explanation of the topic {topic}',
    input_variables=['topic']
)
parser= StrOutputParser()

chain = RunnableSequence(prompt | model | parser)

chain2 = RunnableParallel({
    'joke' : RunnablePassthrough(),
   'explanation' : RunnableSequence(prompt2 | model | parser)
})
chain3 = chain | chain2

result=chain3.invoke({'topic':'air condditioner'})

print(result)