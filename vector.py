from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# load restaurant reviews and initialize embeddings model
df = pd.read_csv("restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# vector database persistence directory
db_location = "./chroma_langchain_db"

# checking if database needs to be populated (first run)
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    # creating Document objects from CSV rows with page_content for vectorization
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"], metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)  # Unique ID using row index
        )

# initializing vector store with embeddings (data that will be stored in the vector db)
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    # adding documents to vector store, and specifying their corresponding unique IDs
    vector_store.add_documents(documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # retrieve top 5 similar documents