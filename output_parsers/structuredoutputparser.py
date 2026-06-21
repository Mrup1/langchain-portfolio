 # structured output parser give us the ability to define the output schema of the model ourself
# it can be used with any model that supports structured output 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser , ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

schema = [
    ResponseSchema(name= 'fact 1 ' , description= 'fact 1 about the topijc'),
    ResponseSchema(name= 'fact 2 ' , description= 'fact 2 about the topijc'),
    ResponseSchema(name= 'fact 3 ' , description= 'fact 3 about the topijc'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='write me 3 fact about the {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)

#disadvantage of the structured output parser is that it doesn not support the data validation 
# meaning if we want three things like name age and country and we want the age to be integer 
#we can not do that with the structured output parser
# it will just give us the output in the format we defined but it will not validate the data 
# so we have to do that manually after getting the output from the model
# so we can use the structured output parser with any model that supports structured output 
# but we have to do the data validation manually after getting the output from the model