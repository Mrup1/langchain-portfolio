from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser  , StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel , Field
from typing import Literal
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel , RunnableBranch , RunnableLambda    

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)
class feedback(BaseModel):
    sentiment:Literal['positive' , 'negative'] = Field(description= 'sentiment will only be positive or negative')

parser = PydanticOutputParser(pydantic_object=feedback)
parser1 = StrOutputParser()

prompt1 = PromptTemplate(
    template='give me the sentiment about the following line \n {feedback}\n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain1 = prompt1 | model | parser

prompt2 = PromptTemplate(
    template = 'write me a appropirate feedback for the positive response \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template = 'write me a appropirate feedback for the negative response \n {feedback}',
    input_variables=['feedback']
)

branch = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not find the sentiment")
)

chain2 = chain1 | branch 

result = chain2.invoke({'feedback': 'this is the a good cellphone'})
chain2.get_graph().print_ascii()
print(result)