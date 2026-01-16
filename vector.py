from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# df(dataframe) will contain restaurant reviews from a CSV file
df = pd.read_csv("restaurant_reviews.csv")