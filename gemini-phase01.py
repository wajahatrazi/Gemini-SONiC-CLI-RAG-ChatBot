import os
from dotenv import find_dotenv, load_dotenv

import json
import PyPDF2

import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# Fetching environment variables from the .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
directory = os.getenv("DIRECTORY") 
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=google_api_key)


#embeddings_model = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/text-embedding-001")


# Read & Extract the data from the directory
"""
def read_files_from_directory(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            with open(os.path.join(directory, filename), 'rb') as file:
                reader = PyPDF2.PdfReader(file)  # Updated from PdfFileReader to PdfReader
                text = ''
                for page_num in range(len(reader.pages)):  # Adjusted for PdfReader
                    text += reader.pages[page_num].extract_text()
                texts.append(text)
        elif filename.endswith('.txt') or filename.endswith('.md'):
            with open(os.path.join(directory, filename), 'r') as file:
                texts.append(file.read())
        elif filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                texts.append(json.dumps(data))
    print("Texts have been extracted")
    return texts
"""

# Updated read_files_from_directory function
def read_files_from_directory(directory):
    texts = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())
                texts.append(text)
        elif filename.endswith(('.txt', '.md')):
            with open(file_path, 'r') as file:
                texts.append(file.read())
        elif filename.endswith('.json'):
            with open(file_path, 'r') as file:
                data = json.load(file)
                texts.append(json.dumps(data))
    print("Texts have been extracted.")
    return texts
    
"""
def get_text_chunks(texts):
    # Flatten the list of texts in case it's a list of lists
    if isinstance(texts[0], list):  # Check if the first element is a list
        texts = [item for sublist in texts for item in sublist]  # Flatten the list
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    chunks = text_splitter.split_text(" ".join(texts))  # Join all texts into a single string
    print("Chunks have been extracted")
    return chunks
"""
# Split texts into chunks
def get_text_chunks(texts):
    # Flatten any nested lists
    texts = [text for sublist in texts for text in (sublist if isinstance(sublist, list) else [sublist])]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    chunks = text_splitter.split_text(" ".join(texts))
    print("Chunks have been created.")
    return chunks

"""
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    print("Vector store has been saved locally")
"""

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("Vector store has been saved locally.")
    

# Main function
def main():
    raw_texts = read_files_from_directory(directory)
    text_chunks_from_files = get_text_chunks(raw_texts)
    get_vector_store(text_chunks_from_files)

if __name__ == "__main__":
    main()
