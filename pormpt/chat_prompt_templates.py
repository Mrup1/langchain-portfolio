from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate([
    ('system', 'you are a good {domin}'),
    ('user', 'tell me about {topic}')
])

print(template.invoke({'domin': 'doctor', 'topic': 'danger of smoking'}))