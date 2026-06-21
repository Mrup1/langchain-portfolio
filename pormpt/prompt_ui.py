from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt

load_dotenv()

model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0)
    
st.header("research assistant")

paper_input = st.selectbox("select the topic for 5 lines research paper",
    ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Natural Language Processing"])
style_input = st.selectbox("select the style",["Formal", "Informal", "Technical", "Conversational"])
length_input = st.selectbox("select the length of the paper", ["Short", "Medium", "Long"])

template = load_prompt("template.json")

#user_input = template.format(paper=paper_input, style=style_input, length=length_input)

if st.button("Submit"):
    chain = template | model
    result = chain.invoke({'paper': paper_input, 'style': style_input, 'length': length_input})

    st.write("Response:")
    st.write(result.content)
  

