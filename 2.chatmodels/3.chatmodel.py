from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
#from langchain_core.messages import HumanMessage # Import HumanMessage for chat

load_dotenv()

llm_huggingface = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", # Highly recommended alternative
    task="text-generation",
    # *** IMPORTANT CHANGE START ***
    # These parameters are now direct arguments to HuggingFaceEndpoint,
    # NOT inside model_kwargs.
    temperature=0.1,
    max_new_tokens=100,
    return_full_text=False, # Often good for chat models to get clean output
    # *** IMPORTANT CHANGE END ***
    timeout=120 # Keep the generous timeout
    # You can remove model_kwargs={} entirely if it's empty, or keep it
    # if you genuinely need to pass other model-specific arguments later.
    # For now, it's not needed.
)

model = ChatHuggingFace(llm=llm_huggingface)

# It's always best practice for chat models to pass a list of message objects
#messages = [
#    HumanMessage(content="what is the capital of india")
#]
result = model.invoke("what is the capital of india") # Pass the list of messages

print(result.content)