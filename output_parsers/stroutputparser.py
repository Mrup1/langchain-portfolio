from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.invoke({'topic': 'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})

result2 = model.invoke(prompt2)

print(result2.content)