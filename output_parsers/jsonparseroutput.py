from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template = 'give me the name , country , and favourt food of shahid afridi \n {format_instructions}',
    input_variables= [],
    partial_variables= {'format_instructions':parser.get_format_instructions()}
)

prompt = template1.invoke({})

chain = template1 | model | parser

result = chain.invoke({})

print(result)

print(type(result))

#we can not define the schema ourself in the json output parser its decided by the model itself