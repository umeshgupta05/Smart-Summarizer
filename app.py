from flask import Flask, request, jsonify, render_template
import whisper
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from flask_cors import CORS  # For handling CORS
from moviepy.editor import VideoFileClip
import tempfile
import shutil

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an 'index.html' in your templates folder

# Step 1: Extract Audio from the Video File
def extract_audio_from_video(video_file):
    try:
        # Use moviepy to extract audio from the uploaded video file
        video_clip = VideoFileClip(video_file)
        audio_file = tempfile.mktemp(suffix=".wav")
        video_clip.audio.write_audiofile(audio_file)
        video_clip.close()
        return audio_file
    except Exception as e:
        raise Exception(f"Error extracting audio from video: {str(e)}")

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
def process_video_file():
    try:
        # Get the video file from the request
        video_file = request.files.get('video_file')

        if not video_file:
            return jsonify({"error": "Video file is required"}), 400

        # Step 1: Save the uploaded video file
        video_path = os.path.join('uploads', video_file.filename)
        video_file.save(video_path)

        # Step 2: Extract audio from the video file
        audio_path = extract_audio_from_video(video_path)

        # Step 3: Convert audio to text using Whisper
        transcript = audio_to_text_whisper(audio_path)

        # Step 4: Summarize the text
        summary = summarize_text(transcript)

        # Clean up the temporary audio file
        os.remove(audio_path)

        # Return the transcript and summary as a JSON response
        return jsonify({
            "transcript": transcript,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs('uploads', exist_ok=True)

    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable
    app.run(host='0.0.0.0', port=port)  # Render will provide the correct port
