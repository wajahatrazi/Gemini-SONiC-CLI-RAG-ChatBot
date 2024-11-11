import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
google_api_key = os.getenv("GOOGLE_API_KEY")

# Load FAISS vector store
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return new_db

vector_store = load_vector_store()

# Initialize Google Gemini model
gemini_model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.1,
    max_output_tokens=500,
)

# Setup a prompt template for the conversation
#prompt_template = """
#As a seasoned SONiC network engineer, answer the question based on the provided context. 
#If the answer is not found, respond with: "Answer is not available in the context."

#Context:
#{context}

#Question: {question}

#Answer:
#"""

prompt_template = """
You are a SONiC Network Engineer with expertise in Operating Systems, Computer Networks, and Datacenter Networks. Answer the question based on the provided context. If the answer is not found in the context, respond with: "I am learning, I do not have the answer in the context."

Context: {context}

Question: {question}

Answer:
"""

# Create a runnable sequence from the prompt and model
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = prompt | gemini_model

# Conversational loop
def sonic_chatbot():
    #history = []

    print("SONiC: I have been developed by Wajahat Razi, a SONiC Engineer at xFlow Research, what can I get you?")

    while True:
        user_input = input("You: ")

        # Convert user input into embeddings
        user_input_embedding = vector_store.embeddings.embed_query(user_input)

        # Use the embeddings for similarity search in the vector store
        docs = vector_store.similarity_search_by_vector(user_input_embedding)

        if not docs:
            print("SONiC: No relevant information found.")
            continue

        # Extract the context from the retrieved documents
        context = "\n".join([doc.page_content for doc in docs])

        # Combine history with the current context
        #full_context = "\n".join([f"User: {entry['parts'][0]}" if entry['role'] == 'user' else f"SONiC: {entry['parts'][0]}" for entry in history]) + f"\nContext:\n{context}"

        # Generate response using the QA chain with full context
        #response = qa_chain.invoke({"context": full_context, "question": user_input})
        response = qa_chain.invoke({"context": context, "question": user_input})

        # Access the content of the response
        model_response = response.content  # Use .content instead of subscript notation

        # Print the response
        print(f"SONiC: {model_response}\n")

        # Append to history
        #history.append({"role": "user", "parts": [user_input]})
        #history.append({"role": "assistant", "parts": [model_response]})

if __name__ == "__main__":
    sonic_chatbot()
