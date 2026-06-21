#RunnableBranch is a control flow component in LangChain that allows you to conditionally route input data to different chains or runnables based on custom logic.
#It functions like an if/elif/else block for chains — where you define a set of condition functions, each associated with a runnable (e.g., LLM call, prompt chain, or tool). The first matching condition is executed. If no condition matches, a default runnable is used (if provided)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnablePassthrough , RunnableBranch , RunnableLambda
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

prompt1 = PromptTemplate(
    template= 'tell a detail explanation the topic {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= 'ssummarize the following text \n {text}',
    input_variables=['text']
)
parser = StrOutputParser()
explanation_chain = RunnableSequence(prompt1 | model | parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500,RunnableSequence(prompt2 | model | parser)),
    (RunnablePassthrough())
)

final_chain = RunnableSequence(explanation_chain , branch_chain)

result = final_chain.invoke({'topic':'data science'})
print(result)