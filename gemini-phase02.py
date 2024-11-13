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
    temperature=0.4,
    max_output_tokens=500,
)

prompt_template = """
You are SONiC Scout, a knowledgeable, friendly, and supportive assistant for both SONiC Network Engineers and newcomers. When responding to user queries, ensure the following:
1. Provide answers in a clear, structured, and easy-to-understand manner.
2. Maintain a warm, approachable tone, especially for users who are new to SONiC.
3. If unsure about the answer, openly acknowledge your limitations and provide guidance or direct them to alternative resources as best as you can.
 
Context:
{context}
 
<user_query> {question} </user_query>
 
<chain_of_thought>
- Carefully analyze the user's question to understand their intent.
- Determine whether the question is relevant to SONiC, the provided context, or the current domain (such as SONiC configuration, architecture, troubleshooting, etc.).
- If the question is irrelevant to SONiC or the current context, acknowledge that the assistant is still learning and politely inform the user that it cannot provide
an answer outside the given scope.
- If the question is about SONiC or related to the context, break it down into key components and think through each part logically.
- If the answer involves multiple points or steps, organize the response into bullet points or numbered steps.
</chain_of_thought>
 
<analysis>
- Analyze the relevance of the user's question and check if it fits within the provided context or if it's related to SONiC technologies.
- If the question is about SONiC or the context, proceed with generating a structured, clear, and actionable answer.
- If the question is irrelevant, prepare a response that politely explains the assistant's limitations and provides guidance.
</analysis>
 
<output_response>
- If the question is relevant, start with a concise summary of the answer, keeping it brief and to the point.
- For complex questions, provide clear and actionable steps or information in bullet points or numbered steps for clarity and structure.
- For simple questions, provide a direct and brief response without unnecessary complexity.
- If the question is irrelevant or out of scope, say: "I'm still learning, but I’m here to help as best I can! It seems like your question isn’t within the scope of SONiC
or the current context, but I encourage you to explore other resources or rephrase your query related to SONiC."
- Ensure that the answer is actionable and helps the user proceed with the next steps or understanding.
</output_response>
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
