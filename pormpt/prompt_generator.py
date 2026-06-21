from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["paper", "style", "length"],
    template="Write a {length} research paper on {paper} in {style} style.",
    validate_template=True
)

template.save("template.json")