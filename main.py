from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2")

# template defining model's role and context for answering questions
template = """

You are expert in answering questions about a pizza restaurant

Here are some relevant reviews about the restaurant: {reviews}

Here is the question to answer: {question}

"""

# creating prompt template and chain to model for processing
prompt = ChatPromptTemplate.from_template(template) 
chain = prompt | model 

# interactive loop for user questions
while True:
    print("\n\n--------------------------------")
    question = input("Ask your question or press q to quit: ")
    print("\n\n")
    if question == "q":
        break

    reviews = retriever.invoke(question)  # retrieving top 5 relevant reviews from vector DB based on the question
    result = chain.invoke({"reviews": reviews, "question": question})  # passing reviews and question to chain
    print(result) 