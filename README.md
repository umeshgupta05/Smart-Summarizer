# 📄 Summarizer & Chatbot

An intelligent Flask-based application that lets you:
- 📝 Upload PDF or 🎥 Video files
- 🔊 Transcribe (for videos) using [Whisper](https://github.com/openai/whisper)
- 🌐 Detect & Translate non-English content to English
- 🧾 Summarize content using BART from Hugging Face
- 💬 Ask questions about the content via a semantic search-based chatbot

---

## 🛠️ Tech Stack

- **Backend**: Flask, HuggingFace Transformers, Whisper, LangChain, FAISS
- **Frontend**: TailwindCSS, jQuery
- **NLP Models**:
  - Summarization: `facebook/bart-large-cnn`
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - ASR: Whisper (`base` model)
- **Translation**: `deep-translator`
- **Language Detection**: `langdetect`

---

## 📂 Features

| Feature            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| 📤 Upload           | Upload `.pdf`, `.mp4`, or `.webm` files                                    |
| 🧠 Auto-Summarize   | Automatically summarize extracted content                                  |
| 🧠 Language Support | Detect and translate non-English text before summarizing                   |
| 💬 Chatbot          | Ask questions based on uploaded content using semantic retrieval            |
| 🧠 Embedding Store  | Store content chunks using FAISS for similarity-based search               |

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/summarizer-chatbot.git
cd summarizer-chatbot


