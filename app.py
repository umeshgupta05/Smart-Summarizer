from flask import Flask, request, jsonify
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from transformers import pipeline
import yt_dlp
import os

app = Flask(__name__)

# IBM Watson API credentials (inbuilt)
IBM_API_KEY = "ZYk7GDnMl1DNKMT1UA3qutttI8-tEIAF0aCmGlAQTq6R"  # Replace with your IBM Watson API key
IBM_URL = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/2a189c18-1d14-4dac-bb14-a634099f9926"

# Step 1: Download YouTube Video
def download_youtube_video(url, output_path="downloaded_video.webm"):
    try:
        ydl_opts = {
            'outtmpl': output_path,  # Specify download location
            'format': 'bestaudio/best',  # Download the best audio-only stream
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

# Step 2: Transcribe Audio to Text using IBM Watson
def audio_to_text_ibm(audio_path):
    try:
        authenticator = IAMAuthenticator(IBM_API_KEY)
        speech_to_text = SpeechToTextV1(authenticator=authenticator)
        speech_to_text.set_service_url(IBM_URL)

        with open(audio_path, "rb") as audio_file:
            response = speech_to_text.recognize(
                audio=audio_file,
                content_type="audio/webm",
                model="en-US_BroadbandModel"
            ).get_result()

        transcript = " ".join(result['alternatives'][0]['transcript'] for result in response['results'])
        return transcript
    except Exception as e:
        raise Exception(f"Error during transcription: {str(e)}")

# Step 3: Summarize the Text
def summarize_text(text):
    try:
        summarizer = pipeline("summarization")
        summary = summarizer(text, max_length=900, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        raise Exception(f"Error during summarization: {str(e)}")

# Flask route for processing the YouTube video
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
        transcript = audio_to_text_ibm(audio_path)

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
