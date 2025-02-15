from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import os
import whisper
from langdetect import detect
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
speech_model = whisper.load_model("base")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = None  # Global variable to store document embeddings

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

def extract_text_from_video(video_path):
    result = speech_model.transcribe(video_path)
    return result["text"]

def detect_and_translate(text):
    lang = detect(text)
    if lang != "en":
        return GoogleTranslator(source=lang, target="en").translate(text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    global vector_store

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    text = ""
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file.filename.endswith((".mp4", ".webm")):
        text = extract_text_from_video(file_path)
    
    text = detect_and_translate(text)
    
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local("faiss_index")

    return jsonify({"summary": summary})

@app.route('/chat', methods=['POST'])
def chatbot():
    global vector_store

    if not vector_store:
        return jsonify({"error": "No document has been uploaded yet!"}), 400

    data = request.get_json()
    question = data.get("question", "")
    retriever = vector_store.as_retriever()
    context = retriever.get_relevant_documents(question)
    best_match = context[0].page_content if context else "I couldn't find relevant information."

    return jsonify({"answer": best_match})

if __name__ == '__main__':
    app.run(debug=True)
