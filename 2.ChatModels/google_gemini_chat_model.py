from langchain_google_genai import ChatGoogleGenerativeAI as ChatG
from dotenv import load_dotenv

load_dotenv()

model = ChatG(model = 'gemini-2.5-flash')
result = model.invoke("What is capital of pakstan?")
print(result.content)