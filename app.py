from flask import Flask, request, jsonify, render_template_string
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
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        doc.close()  # Properly close the document
        return text
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def extract_text_from_video(video_path):
    try:
        result = speech_model.transcribe(video_path)
        return result["text"]
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source=lang, target="en").translate(text)
        return text
    except Exception as e:
        raise Exception(f"Error during translation: {str(e)}")

def clean_text(text):
    # Add basic text cleaning
    return ' '.join(text.split())

@app.route('/')
def home():
    # Use render_template_string since we have the HTML content
    return render_template_string(HTML_CONTENT)  # HTML_CONTENT would be your index.html content

@app.route('/process', methods=['POST'])
def process_file():
    global vector_store
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    allowed_extensions = {'.pdf', '.mp4', '.webm'}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # Extract text based on file type
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_video(file_path)

        # Clean and translate text
        text = clean_text(text)
        text = detect_and_translate(text)

        # Generate summary
        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']

        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store = FAISS.from_documents(documents, embedding_model)
        
        # Save vector store
        os.makedirs("faiss_index", exist_ok=True)
        vector_store.save_local("faiss_index")

        # Clean up uploaded file
        os.remove(file_path)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chatbot():
    global vector_store
    
    if not vector_store:
        return jsonify({"error": "No document has been processed yet"}), 400

    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data["question"]
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        contexts = retriever.get_relevant_documents(question)
        
        if not contexts:
            return jsonify({"answer": "I couldn't find relevant information in the document."}), 200

        # Combine relevant contexts
        combined_context = " ".join([doc.page_content for doc in contexts])
        
        return jsonify({"answer": combined_context})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
