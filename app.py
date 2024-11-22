from flask import Flask, request, jsonify, render_template
import whisper
from transformers import T5ForConditionalGeneration, T5Tokenizer
import yt_dlp
import os
import shutil

from flask_cors import CORS  # For handling CORS

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

@app.route('/')

def home():
    return render_template('index.html')  # Ensure you have an 'index.html' in your templates folder

# Step 1: Download YouTube Video
def download_youtube_video(url, output_path="downloaded_video.webm"):
    try:
        ydl_opts = {
            'outtmpl': output_path,  # Specify download location
            'format': 'bestaudio/best',  # Download the best audio-only stream
            'noplaylist': True,  # Ensure only the specified video is downloaded
            'quiet': True,  # Suppress the output to make it cleaner
        }

        # If authentication is needed, you can specify cookies or credentials directly
        # ydl_opts['cookiefile'] = 'cookies.txt'  # Use a cookies file if needed for access

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return output_path
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

# Step 2: Transcribe Audio to Text using Whisper
def audio_to_text_whisper(audio_path):
    try:
        # Load the Whisper model (use 'base', 'small', 'medium', or 'large' depending on your system's capabilities)
        model = whisper.load_model("base")  # Replace "base" with a larger model if needed
        
        # Transcribe the audio file
        result = model.transcribe(audio_path)
        
        # Extract the transcript from the result
        transcript = result['text']
        return transcript
    except Exception as e:
        raise Exception(f"Error during transcription with Whisper: {str(e)}")

# Step 3: Summarize the Text using T5
def summarize_text(text):
    try:
        # Load the T5 model and tokenizer
        model_name = "t5-base"  # You can also use "t5-base" or "t5-large" for better results
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Prepare the input text for summarization
        input_text = f"summarize: {text}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate the summary
        summary_ids = model.generate(
            inputs, 
            max_length=150,  # Adjust the maximum length of the summary
            min_length=30,   # Adjust the minimum length of the summary
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    except Exception as e:
        raise Exception(f"Error during summarization: {str(e)}")

@app.route('/process', methods=['POST'])
def process_youtube_video():
    try:
        # Get the YouTube URL from the request
        data = request.get_json()
        youtube_url = data.get('youtube_url')

        if not youtube_url:
            return jsonify({"error": "YouTube URL is required"}), 400

        # Step 1: Download YouTube video
        audio_path = download_youtube_video(youtube_url)

        # Step 2: Convert audio to text
        transcript = audio_to_text_whisper(audio_path)

        # Step 3: Summarize the text
        summary = summarize_text(transcript)

        # Return the summary as a JSON response
        return jsonify({
            "transcript": transcript,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable
    app.run(host='0.0.0.0', port=port)  # Render will provide the correct port
