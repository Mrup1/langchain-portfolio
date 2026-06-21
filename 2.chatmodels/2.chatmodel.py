from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()       

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2)

user_input = input("Enter your prompt: ")
result = model.invoke(user_input)

print(result.content)   
