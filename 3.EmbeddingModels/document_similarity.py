from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian cricket team captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Who is former captain of the Indian cricket team?"
doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarity_score = cosine_similarity([query_embedding], doc_embeddings)[0]

best_match_index = np.argmax(similarity_score)

print(f"Best matching document: {documents[best_match_index]}")
