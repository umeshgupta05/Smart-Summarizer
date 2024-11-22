from flask import Flask, request, jsonify, render_template
import whisper
from transformers import T5ForConditionalGeneration, T5Tokenizer
import yt_dlp
import os
import shutil

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Whisper model
model = whisper.load_model("base")

# Step 1: Download YouTube Video
def download_youtube_video(url, output_path="downloaded_video.webm"):
    try:
        ydl_opts = {
            'outtmpl': output_path,  # Specify download location
            'format': 'bestaudio/best',  # Download the best audio-only stream
            'noplaylist': True,  # Ensure only the specified video is downloaded
            'quiet': True,  # Suppress the output to make it cleaner
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return output_path
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

# Step 2: Transcribe Audio to Text using Whisper
def audio_to_text_whisper(audio_path):
    try:
        result = model.transcribe(audio_path)
        transcript = result['text']
        return transcript
    except Exception as e:
        raise Exception(f"Error during transcription with Whisper: {str(e)}")

# Step 3: Summarize the Text using T5
def summarize_text(text):
    try:
        model_name = "t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        input_text = f"summarize: {text}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        summary_ids = model.generate(
            inputs,
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    except Exception as e:
        raise Exception(f"Error during summarization: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video_or_url():
    try:
        data = request.form
        youtube_url = data.get('youtube_url')
        file = request.files.get('video_file')

        if youtube_url:
            # Process YouTube URL
            audio_path = download_youtube_video(youtube_url)
        elif file:
            # Process uploaded video file
            audio_path = "uploaded_video.mp4"
            file.save(audio_path)
        else:
            return jsonify({"error": "No YouTube URL or video file provided"}), 400

        # Convert audio to text
        transcript = audio_to_text_whisper(audio_path)

        # Summarize the text
        summary = summarize_text(transcript)

        # Return the summary as a JSON response
        return jsonify({
            "transcript": transcript,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
