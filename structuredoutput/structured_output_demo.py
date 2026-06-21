from langchain_huggingface  import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict , Annotated , Literal

load_dotenv()

model = HuggingFaceEndpoint(repo_id="HuggingFaceTB/SmolLM3-3B",
                            task = "text generation",
                            temperature=0.1)
model = ChatHuggingFace(llm=model)

class structured_review(TypedDict):
    #summary: Annotated[str, "A brief summary of the movie"]
    rating: Annotated[float, "Rating out of 10"]
    director: Annotated[Literal['me' , 'you'],"choose between me or you"]

structured_model = model.with_structured_output(structured_review)

result = structured_model.invoke('write the summary of movie inception')

print(result)