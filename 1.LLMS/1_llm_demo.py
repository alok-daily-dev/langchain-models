from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
result = llm.invoke("What is capital of pakstan?")

print(result.content)