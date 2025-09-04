from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

documents = [
    "What is the capital of Pakistan?",
    "Who is the president of the USA?",
]

embeddings_list = embeddings.embed_documents(documents)

print(str(embeddings_list))