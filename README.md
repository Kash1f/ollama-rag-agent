# Restaurant Reviews Retriever with Vector DB

This project demonstrates how to build a **retrieval-augmented system** using a CSV of restaurant reviews, embeddings, and a vector database to enable context-aware responses from an LLM.

## Features

* Converts restaurant reviews from CSV into **vector embeddings** using `OllamaEmbeddings`.
* Stores embeddings in a **persistent Chroma vector database** for fast semantic search.
* Retrieves **top relevant reviews** for any query using a retriever.
* Integrates with an LLM chain to generate **contextually informed responses**.

## Installation

```bash
pip install pandas langchain chromadb ollama
```

## Usage

1. Place your CSV file (`restaurant_reviews.csv`) in the project directory.
2. Run the script to populate the vector database (first run only):

```bash
python main.py
```

3. Use the retriever to query relevant reviews:

```python
reviews = retriever.invoke("Your question here")
# Pass reviews to your LLM chain for response
```

Subsequent runs will **reuse the vector database**, so the embedding step does not need to be repeated.

## Project Structure

* `restaurant_reviews.csv` — source data of reviews
* `main.py` — script for generating embeddings, populating the vector DB, and querying
* `chroma_langchain_db/` — persistent directory for the vector database

## Notes

* Each row in the CSV is converted into a `Document` object with a unique ID, page content, and metadata.
* The retriever automatically performs semantic search using similarity to the query.

## License

MIT
