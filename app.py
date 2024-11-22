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
from pydub import AudioSegment
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

nltk.download("punkt_tab")

IBM_API_KEY = "ZYk7GDnMl1DNKMT1UA3qutttI8-tEIAF0aCmGlAQTq6R"  # Replace with your IBM Watson API key
IBM_URL = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/2a189c18-1d14-4dac-bb14-a634099f9926"

ALLOWED_EXTENSIONS = {'pdf', 'webm', 'mp4', 'wav'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert audio files to WAV format
def convert_to_wav(input_path, output_path="converted_audio.wav"):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        raise Exception(f"Error converting audio file to WAV: {str(e)}")

# IBM Watson transcription
def audio_to_text_ibm(audio_path):
    try:
        authenticator = IAMAuthenticator(IBM_API_KEY)
        speech_to_text = SpeechToTextV1(authenticator=authenticator)
        speech_to_text.set_service_url(IBM_URL)

        with open(audio_path, "rb") as audio_file:
            response = speech_to_text.recognize(
                audio=audio_file,
                content_type="audio/wav",  # Use 'audio/webm' for webm files
                model="en-US_BroadbandModel"
            ).get_result()

        transcript = " ".join(result['alternatives'][0]['transcript'] for result in response['results'])
        return transcript
    except Exception as e:
        raise Exception(f"IBM Watson Transcription Error: {str(e)}")

# Text summarization
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

# PDF to text
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

# Download YouTube video
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

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists in the templates folder

@app.route('/process', methods=['POST'])
def process_input():
    try:
        uploads_dir = os.path.join(os.getcwd(), "uploads")
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)  # Create uploads directory if not exists

        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(uploads_dir, filename)
                file.save(file_path)

                if filename.endswith(".pdf"):
                    # Process PDF
                    text = pdf_to_text(file_path)
                    summary = summarize_text(text)
                else:
                    # Convert to WAV for audio transcription
                    wav_path = convert_to_wav(file_path)
                    transcript = audio_to_text_ibm(wav_path)
                    summary = summarize_text(transcript)

                os.remove(file_path)  # Cleanup original file
                return jsonify({"summary": summary})

        # Handle YouTube URL
        data = request.get_json()
        youtube_url = data.get('youtube_url')
        if youtube_url:
            audio_path = download_youtube_video(youtube_url)
            wav_path = convert_to_wav(audio_path)
            transcript = audio_to_text_ibm(wav_path)
            summary = summarize_text(transcript)

            os.remove(audio_path)  # Cleanup downloaded video
            os.remove(wav_path)    # Cleanup converted audio
            return jsonify({"summary": summary})

        return jsonify({"error": "No valid input provided"}), 400

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
