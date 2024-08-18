import os
import fitz  # PyMuPDF
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from prompt import chatbot_prompt

openai_api_key = ""

app = Flask(__name__)
client = OpenAI(api_key=openai_api_key)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text


# Function to extract text from all PDFs in a folder
def extract_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            texts.append(text)
    return texts


# Function to create a vector database using FAISS
def create_vector_database(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray()
    
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    
    return index, vectorizer


# Function to find the most relevant text using FAISS
def find_most_relevant_text(query, index, vectorizer, texts, k=4):
    query_vector = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vector, k=k)

    # Retrieve the top k relevant texts
    relevant_texts = [texts[idx] for idx in indices[0]]
    return relevant_texts




# Function to get answers from OpenAI API
def get_answer_from_openai(query, relevant_texts):
    relevant_text_combined = "\n\n".join(relevant_texts)
    messages = [
        {"role": "system", "content": chatbot_prompt},
        {"role": "user", "content": f"Context: {relevant_text_combined}\n\nQuestion: {query}\nAnswer:"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=350
    )
    return response.choices[0].message.content


# Extract texts and create vector database (this part only needs to be done once)
folder_path = '/Users/lukegeel/Desktop/RentwisePDFs'  # Replace with the path to your folder containing PDFs
texts = extract_texts_from_folder(folder_path)
index, vectorizer = create_vector_database(texts)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['question']
    try:
        # Find the most relevant text using the RAG pipeline
        relevant_texts = find_most_relevant_text(user_input, index, vectorizer, texts)
        # Get the answer from OpenAI
        answer = get_answer_from_openai(user_input, relevant_texts)
        return jsonify({"answer": answer})
    except Exception as e:
        app.logger.error(f"Failed to fetch response from OpenAI: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
