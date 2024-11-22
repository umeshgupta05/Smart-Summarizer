from flask import Flask, request, jsonify, render_template
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from collections import Counter
import heapq
import yt_dlp
import PyPDF2
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

nltk.download("punkt_tab")
# IBM Watson API credentials
IBM_API_KEY = "ZYk7GDnMl1DNKMT1UA3qutttI8-tEIAF0aCmGlAQTq6R"  # Replace with your IBM Watson API key
IBM_URL = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/2a189c18-1d14-4dac-bb14-a634099f9926"

ALLOWED_EXTENSIONS = {'pdf', 'webm', 'mp4', 'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an 'index.html' in your templates folder

# Step 1: Download YouTube Video
def download_youtube_video(url, output_path="downloaded_video.webm"):
    try:
        ydl_opts = {
            'outtmpl': output_path,
            'format': 'bestaudio/best',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

# Step 2: Transcribe Audio using IBM Watson
def audio_to_text_ibm(audio_path):
    try:
        authenticator = IAMAuthenticator(IBM_API_KEY)
        speech_to_text = SpeechToTextV1(authenticator=authenticator)
        speech_to_text.set_service_url(IBM_URL)

        with open(audio_path, "rb") as audio_file:
            response = speech_to_text.recognize(
                audio=audio_file,
                content_type="audio/webm",  # Adjust format if using mp4
                model="en-US_BroadbandModel"
            ).get_result()

        transcript = " ".join(result['alternatives'][0]['transcript'] for result in response['results'])
        return transcript
    except Exception as e:
        raise Exception(f"Error during transcription: {str(e)}")

def summarize_text(text):
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        word_frequencies = Counter(words)

        # Calculate scores for sentences
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

        # Extract top 3 sentences
        summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
        return " ".join(summary_sentences)
    except Exception as e:
        raise Exception(f"Error during summarization: {str(e)}")

# Process PDF to Text
def pdf_to_text(pdf_path):
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

@app.route('/process', methods=['POST'])
def process_input():
    try:
        # Check if the request has a file
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join("uploads", filename)
                file.save(file_path)

                if filename.endswith(".pdf"):
                    # Process PDF
                    text = pdf_to_text(file_path)
                    summary = summarize_text(text)
                else:
                    # Process Video
                    transcript = audio_to_text_ibm(file_path)
                    summary = summarize_text(transcript)
                
                os.remove(file_path)  # Cleanup
                return jsonify({"summary": summary})

        # Check if a YouTube URL is provided
        data = request.get_json()
        youtube_url = data.get('youtube_url')
        if youtube_url:
            audio_path = download_youtube_video(youtube_url)
            transcript = audio_to_text_ibm(audio_path)
            summary = summarize_text(transcript)
            os.remove(audio_path)  # Cleanup
            return jsonify({"summary": summary})

        return jsonify({"error": "No valid input provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")  # Create upload directory if it doesn't exist
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
