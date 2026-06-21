from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage
from dotenv import load_dotenv

load_dotenv() 
model = HuggingFaceEndpoint(repo_id="HuggingFaceTB/SmolLM3-3B",
                        task = "text generation",
                        temperature=0.1)
model = ChatHuggingFace(llm=model)

chat_history = [SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("Enter your query: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print("ai: ", result.content)

print("Chat History:" , chat_history)