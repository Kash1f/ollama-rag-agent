from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate



# Llama should be running in the bg on your computer and it's going to expose a server or like an HTTP REST API that will communicate with our program(code). So when you pull these models, they are actually running locally on your own computer, and we can trigger Ollama to utilize these models from code in Python, or we can do it diretly from the cmd.

# We will utlize the Ollama model, we will specify the model that I want to use from Ollama, then we can start utilizing the model and kind of invoking it.

model = OllamaLLM(model="llama3.2")

# Inside the string we will specify what the model will actually do
template = """

You are expert in answering questions about a pizza restaurant

Here are some relevant reviews about the restaurant: {reviews}

Here is the question to answer: {question}

"""

# in the chat prompt template we will pass in a reviews variable and a question variable and then the model can respond to that

prompt = ChatPromptTemplate.from_template(template) 
chain = prompt | model # we are chaining the prompt with the model so that when we invoke the chain, it will first format the prompt (with the reviews and question) and then pass it to the model to get a response

result = chain.invoke({"reviews": [], "question": "What is the best pizza place in this town?"})
print(result) # we are invoking the chain with an empty list of reviews and a question about the best pizza place in town. The model will then respond based on the prompt we provided earlier.