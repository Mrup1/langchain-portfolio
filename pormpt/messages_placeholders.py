from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

template = ChatPromptTemplate([
    ('system', 'you are a good assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user','{query}')
])

chat_history = []

with open ('placeholder.txt') as file:
    chat_history.extend(file.readlines())

#print(chat_history)

print(template.invoke({'chat_history': chat_history, 'query': 'tell me about the weather today'}))