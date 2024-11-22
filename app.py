from flask import Flask, request, jsonify, render_template
import yt_dlp
import os
from flask_cors import CORS  # For handling CORS

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an 'index.html' in your templates folder

# Step 1: Extract YouTube Captions (as string)
def get_youtube_captions(url, language='en'):
    try:
        ydl_opts = {
            'quiet': True,  # Suppress the output to make it cleaner
            'format': 'bestaudio/best',  # Download the best audio-only stream
            'noplaylist': True,  # Ensure only the specified video is downloaded
            'writesubtitles': True,  # Download subtitles (captions)
            'subtitleslangs': [language],  # Language of the subtitles (e.g., 'en' for English)
            'writeautomaticsub': True,  # Include automatic captions if available
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)
            subtitle_file = None
            # Check if captions are available and select the appropriate subtitle file
            if 'subtitles' in result and language in result['subtitles']:
                subtitle_file = result['subtitles'][language][0]['url']
            if not subtitle_file:
                raise Exception(f"No subtitles available for this video in {language}.")

            # Fetch the subtitle file content directly
            import requests
            response = requests.get(subtitle_file)
            if response.status_code == 200:
                captions = response.text
                # Clean up the VTT format to extract text only
                cleaned_text = ' '.join(line.strip() for line in captions.split('\n') if not line.startswith('NOTE') and line.strip())
                return cleaned_text
            else:
                raise Exception(f"Failed to fetch captions, status code: {response.status_code}")
    except Exception as e:
        raise Exception(f"Error extracting captions: {str(e)}")

# Step 2: Summarize the Text using T5
def summarize_text(text):
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
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

        # Step 1: Extract YouTube video captions directly (as string)
        transcript = get_youtube_captions(youtube_url)

        # Step 2: Summarize the text
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
