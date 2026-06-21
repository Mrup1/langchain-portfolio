#Why Use PydanticOutputParser ?
#Strict Schema Enforcement - Ensures that LLM responses follow a well-defined structure.
#Type Safety - Automatically converts LLM outputs into Python object
#Data Validation - Validates the data against the Pydantic model, ensuring correctness.we can define the data types ourself 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser 
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

class student( BaseModel):
    name: str  #  default value
    age: int = Field(default=20, ge=18, le=30, description="Age must be between 18 and 30") # ge = greater than or equal to , le = less than or equal to
    gpa: float = Field(default=3.4 , gt = 0 , lt = 5 , description="GPA must be between 0 and 5")

parser = PydanticOutputParser(pydantic_object=student)

template = PromptTemplate(
    template = 'give me the name , age , and gpa of a {university}student \n {format_instructions}',
    input_variables= ['university'],
    partial_variables= {'format_instructions':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'university': 'fast university'})

print(result)

# stringoutput parser is used to get the output in string format
# jsonoutput parser is used to get the output in json format doesnot porvide the schema model itself make the schema
# structuredoutput parser is used to get the output in structured format we can define the schema our self but doesnot porvide the data validation
# pydanticoutput parser is used to get the output in structured format we can define the schema our self and it also provides the data validation