from flask import Flask, request, jsonify, render_template
import whisper
import os
import requests

# Initialize the Flask app

app = Flask(__name__, template_folder="templates")
# Load the Whisper model (base model; change to your desired size if needed)
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded successfully!")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    custom_prompt = request.form.get('prompt', '')

    # Determine the file extension
    file_extension = os.path.splitext(audio_file.filename)[1]
    temp_file_path = f"temp_audio{file_extension}"

    # Save the file temporarily
    audio_file.save(temp_file_path)
    try:
        # Transcribe the audio using Whisper
        result = model.transcribe(temp_file_path)
        transcription = result["text"]
        prompt = custom_prompt + "\nHere is the transcription: " + transcription
        response = requests.post(
            "http://localhost:11434/api/generate",  # Target local server
            json={
                "model": "qwen2.5",
                "prompt": prompt,
                "stream": False
            }  # Send transcription as JSON
        )

        # Extract the "response" key from the JSON response
        if response.status_code == 200:
            print(response.json())
            response_json = response.json()
            if "response" in response_json:
                return jsonify({"response": response_json["response"]}), 200
            else:
                return jsonify({"error": "'response' key not found in forwarded server response"}), 500
        else:
            return jsonify({"error": "Failed to forward transcription", "details": response.text}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# Create an endpoint for audio transcription
# @app.route('/transcribe', methods=['POST'])
# def transcribe():
#     # # Check if an audio file is part of the request
#     # if 'audio' not in request.files:
#     #     return jsonify({"error": "No audio file provided"}), 400

#     # audio_file = request.files['audio']

#     # # Save the file temporarily
#     # temp_file_path = "temp_audio.wav"
#     # audio_file.save(temp_file_path)

#     # try:
#     #     # Transcribe the audio using Whisper
#     #     result = model.transcribe(temp_file_path)
#     #     transcription = result["text"]
#     #     return jsonify({"transcription": transcription}), 200
#         # Send the transcription to localhost:11435
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file provided"}), 400

#     audio_file = request.files['audio']

#     # Save the file temporarily
#     temp_file_path = "temp_audio.wav"
#     audio_file.save(temp_file_path)
#     try:
#         result = model.transcribe(temp_file_path)
#         transcription = result["text"]
#         response = requests.post(
#             "http://localhost:11435/api/generate",  # Target local server
#             json={
                
#                 "model": "qwen2.5",
#                 "prompt": """You are an AI assistant at an ophthalmology clinic in Australia. 
#                         You are given a transcription of a user's voice with a patient talking as well. 
#                         Your task is to provide a summarised account of the visit. 
#                         Here is the transcription: """ + transcription,
#                 "stream": False
                
#                 }  # Send transcription as JSON
#         )

#         # Extract the "response" key from the JSON response
#         if response.status_code == 200:
#             response_json = response.json()
#             if "response" in response_json:
#                 return jsonify({"response": response_json["response"]}), 200
#             else:
#                 return jsonify({"error": "'response' key not found in forwarded server response"}), 500
#         else:
#             return jsonify({"error": "Failed to forward transcription", "details": response.text}), response.status_code

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     finally:
#         # Clean up the temporary file
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)

@app.route('/')
def home():
    return render_template('audio_record.html')


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=("cert.pem", "key.pem"))
