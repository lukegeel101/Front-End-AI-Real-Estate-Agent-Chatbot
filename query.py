import os
import fitz  # PyMuPDF
import faiss
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    print("extracted text from pdf")
    return text

# Function to extract text from all PDFs in a folder
def extract_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            texts.append(text)
            print("extracted text from folder")
    return texts

# Initialize the OpenAI API
openai.api_key = ''  # Replace with your OpenAI API key
client = OpenAI(api_key='')

# Function to create a vector database using FAISS
def create_vector_database(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray()
    
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    print("created vector db")
    
    return index, vectorizer

# Function to find the most relevant text using FAISS
def find_most_relevant_text(query, index, vectorizer, texts, k=4):
    query_vector = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vector, k=k)

    # Retrieve the top k relevant texts
    relevant_texts = [texts[idx] for idx in indices[0]]
    print("found most relevant pdf")
    return relevant_texts

# Function to get answers from OpenAI API
def get_answer_from_openai(query, relevant_texts):
    relevant_text_combined = "\n\n".join(relevant_texts)
    messages = [
        {"role": "system", "content": "You are a helpful and friendly real estate agent."},
        {"role": "user", "content": f"Context: {relevant_text_combined}\n\nQuestion: {query}\nAnswer:"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content

# Example usage
folder_path = '/Users/lukegeel/Desktop/...'  # Replace with the path to your folder containing PDFs
query = 'Show me the cheapest studio units'  # Replace with your query

# Extract texts and create vector database (this part only needs to be done once)
texts = extract_texts_from_folder(folder_path)
index, vectorizer = create_vector_database(texts)

# Reuse the vector database and vectorizer for new queries
relevant_texts = find_most_relevant_text(query, index, vectorizer, texts)
answer = get_answer_from_openai(query, relevant_texts)
print(answer)
