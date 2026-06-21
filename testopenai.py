import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0 , api_key=os.getenv("OPEN_AI_KEY"))

print(model.invoke("What is the capital of France?").content)
#print(os.getenv("OPEN_AI_KEY"))

