from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import whisper
import os
import requests
from utils.database import init_db, get_next_version
import sqlite3
import logging
import markdown
import time
from flask import g
import subprocess
import traceback
import tempfile
from queue import Queue, Empty
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor
import atexit
import queue
import threading
import concurrent.futures



# Initialize the Flask app
app = Flask(__name__, template_folder="templates")
app.jinja_env.filters['markdown'] = lambda text: markdown.markdown(text) if text else ''
init_db()
# Load the Whisper model (base model; change to your desired size if needed)
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded successfully!")

# Memory Management
MAX_STORED_CHUNTS = 10  # Only keep last N chunks in memory

# Global queue and processing settings
CHUNK_QUEUE = Queue()
MAX_WORKERS = 3  # Number of concurrent transcription workers
SHUTDOWN_EVENT = Event()

# Initialize thread pool for transcription processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Error handling for misaligned chunks
def handle_misaligned_chunks(chunks):
    # If overlap detection fails, fall back to timestamp-based alignment
    sorted_chunks = sorted(chunks, key=lambda x: x['time'])
    return ' '.join(chunk['text'] for chunk in sorted_chunks)

# Performance s
def monitor_chunk_processing(start_time, chunk_size):
    processing_time = time.time() - start_time
    if processing_time > chunk_size * 0.8:  # If processing takes >80% of chunk time
        log.warning(f"Chunk processing too slow: {processing_time}s for {chunk_size}s chunk")

# Handle different speakers
def detect_speaker_changes(transcription):
    # Whisper can detect speaker changes
    # We could use this to better segment the chunks
    speaker_segments = transcription.get('speaker_segments', [])
    return speaker_segments

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    hash_value = request.form.get('hash')
    doctor = request.form.get('doctor')
    custom_prompt = request.form.get('prompt', '')
    selected_model = request.form.get('model', 'qwen2.5')
    save_audio = request.form.get('save_audio', 'false').lower() == 'true'

    if not hash_value or not doctor:
        return jsonify({"error": "Missing hash or doctor"}), 400

    file_extension = os.path.splitext(audio_file.filename)[1]
    temp_file_path = f"temp_audio{file_extension}"

    try:
        audio_file.save(temp_file_path)
        result = model.transcribe(temp_file_path)
        transcription = result["text"]
        
        # Save transcript immediately
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        version = get_next_version(hash_value, doctor)
        
        cursor.execute('''
            INSERT INTO transcriptions 
            (hash, doctor, version, prompt_used, raw_transcript, summary, model_used, audio_saved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (hash_value, doctor, version, custom_prompt, transcription, "Processing...", selected_model, save_audio))
        
        conn.commit()

        # Save audio file if requested
        if save_audio:
            audio_dir = os.path.join('audio_files', hash_value, doctor)
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f'recording_v{version}{file_extension}')
            audio_file.save(audio_path)
        
        # Now try to get the summary
        prompt = custom_prompt + "\nHere is the transcription: " + transcription
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": selected_model,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            response_json = response.json()
            if "response" in response_json:
                summary = response_json["response"]
                
                # Update the summary
                cursor.execute('''
                    UPDATE transcriptions 
                    SET summary = ?
                    WHERE hash = ? AND doctor = ? AND version = ?
                ''', (summary, hash_value, doctor, version))
                
                conn.commit()
                conn.close()

                return jsonify({
                    "response": summary,
                    "version": version,
                    "raw_transcript": transcription
                }), 200
            
            return jsonify({
                "error": "Response key not found",
                "version": version,
                "raw_transcript": transcription
            }), 500
        
        return jsonify({
            "error": "Failed to generate summary",
            "version": version,
            "raw_transcript": transcription
        }), response.status_code

    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'conn' in locals():
            conn.close()

@app.route('/')
def home():
    hash_value = request.args.get('hash')
    doctor = request.args.get('doctor')

    if not hash_value or not doctor:
        return "Missing required parameters", 400

    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()

    # Fetch available LLM models
    cursor.execute('SELECT name, is_default FROM llm_models WHERE is_active = TRUE')
    llm_models = cursor.fetchall()

    # Fetch prompts (both global and doctor-specific)
    cursor.execute('''
        SELECT prompt_text, is_default, id 
        FROM prompts 
        WHERE doctor_id = ? OR doctor_id = '~All'
        ORDER BY 
            CASE 
                WHEN doctor_id = ? THEN 0 
                ELSE 1 
            END,
            is_default DESC,
            priority DESC
    ''', (doctor, doctor))
    prompts = cursor.fetchall()

    # Fetch previous recordings
    cursor.execute('''
        SELECT version, prompt_used, raw_transcript, summary, model_used, created_at, audio_saved
        FROM transcriptions 
        WHERE hash = ? AND doctor = ?
        ORDER BY version DESC
    ''', (hash_value, doctor))
    previous_recordings = cursor.fetchall()
    conn.close()

    next_version_number = 1 if not previous_recordings else previous_recordings[0][0] + 1

    return render_template('audio_record.html', 
                         hash_value=hash_value,
                         doctor=doctor,
                         llm_models=llm_models,
                         prompts=prompts,
                         previous_recordings=previous_recordings,
                         next_version_number=next_version_number)

@app.route('/retry_summary', methods=['POST'])
def retry_summary():
    hash_value = request.form.get('hash')
    doctor = request.form.get('doctor')
    custom_prompt = request.form.get('prompt', '')
    selected_model = request.form.get('model', 'qwen2.5')
    transcription = request.form.get('raw_transcript', '')

    if not all([hash_value, doctor, transcription]):
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        # Save new version with the existing transcript
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        version = get_next_version(hash_value, doctor)
        
        # Try to get the summary
        prompt = custom_prompt + "\nHere is the transcription: " + transcription
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": selected_model,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            response_json = response.json()
            if "response" in response_json:
                summary = response_json["response"]
                
                cursor.execute('''
                    INSERT INTO transcriptions 
                    (hash, doctor, version, prompt_used, raw_transcript, summary, model_used, audio_saved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (hash_value, doctor, version, custom_prompt, transcription, summary, selected_model, False))
                
                conn.commit()
                conn.close()

                return jsonify({
                    "response": summary,
                    "version": version,
                    "raw_transcript": transcription
                }), 200

        return jsonify({
            "error": "Failed to generate summary",
            "version": version,
            "raw_transcript": transcription
        }), response.status_code

    except Exception as e:
        logging.error(f"Retry summary error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@app.route('/save_prompt', methods=['POST'])
def save_prompt():
    doctor = request.form.get('doctor')
    prompt_text = request.form.get('prompt_text')
    is_default = request.form.get('is_default', 'false').lower() == 'true'
    
    if not doctor or not prompt_text:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prompts (doctor_id, prompt_text, is_default)
            VALUES (?, ?, ?)
        ''', (doctor, prompt_text, is_default))
        
        conn.commit()
        conn.close()
        
        return jsonify({"message": "Prompt saved successfully"}), 200
    except sqlite3.Error as e:
        logging.error(f"Error saving prompt: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe_chunk', methods=['POST'])
def transcribe_chunk():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        chunk_number = request.form.get('chunk_number', '0')
        hash_value = request.form.get('hash')
        doctor = request.form.get('doctor')
        version = request.form.get('version', '0')
        
        if not hash_value or not doctor:
            return jsonify({"error": "Missing hash or doctor"}), 400
        
        # Create debug directory
        debug_dir = os.path.join('debug_audio', hash_value, doctor, version, chunk_number)
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save paths
        audio_path = os.path.join(debug_dir, 'original.webm')
        wav_path = os.path.join(debug_dir, 'converted.wav')
        
        # Save original audio
        audio_file.save(audio_path)
        logging.debug(f"Saved chunk {chunk_number} to {audio_path}")

        # Submit transcription task to thread pool
        future = executor.submit(
            process_transcription,
            audio_path,
            wav_path,
            chunk_number,
            hash_value,
            doctor,
            version
        )

        try:
            # Wait for result with timeout
            transcription = future.result(timeout=30)
            transcription = clean_transcription(transcription)
            print(f"*****************Transcription: {transcription}")
            print(f"Chunk number: {chunk_number}")
            # Save transcription for debugging
            with open(os.path.join(debug_dir, 'transcription.txt'), 'w') as f:
                f.write(transcription)
            
            # Save to database
            conn = sqlite3.connect('app.db')
            cursor = conn.cursor()
            
            # Save chunk transcription
            cursor.execute('''
                INSERT INTO chunk_transcriptions 
                (hash, doctor, version, chunk_number, transcription)
                VALUES (?, ?, ?, ?, ?)
            ''', (hash_value, doctor, version, chunk_number, transcription))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                "transcription": transcription,
                "chunk_number": chunk_number
            }), 200
            
        except concurrent.futures.TimeoutError:
            logging.error(f"Transcription timeout for chunk {chunk_number}")
            return jsonify({"error": "Transcription timeout"}), 504

    except Exception as e:
        logging.error(f"Error handling chunk {chunk_number}: {str(e)}")
        return jsonify({"error": str(e)}), 500

def process_queue_worker():
    """Worker that processes chunks from the queue"""
    while not SHUTDOWN_EVENT.is_set():
        try:
            # Get item from queue with timeout to allow checking shutdown event
            item = CHUNK_QUEUE.get(timeout=1)
            
            try:
                # Process the transcription
                transcription = process_transcription(**item)
                
                # Update the client (you might want to implement a websocket or SSE for this)
                if item.get('callback'):
                    item['callback'](transcription)
                
            except Exception as e:
                print(f"Error processing queued chunk: {e}")
                traceback.print_exc()
            
            finally:
                CHUNK_QUEUE.task_done()
                
        except Empty:
            continue
        except Exception as e:
            print(f"Error in queue worker: {e}")
            traceback.print_exc()

# Start queue processing workers
for _ in range(MAX_WORKERS):
    worker = Thread(target=process_queue_worker, daemon=True)
    worker.start()

@app.route('/generate_summary', methods=['POST'])

def generate_summary():
    try:
        hash_value = request.form.get('hash')
        doctor = request.form.get('doctor')
        prompt = request.form.get('prompt')
        model_name = request.form.get('model')
        
        # Get current version
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COALESCE(MAX(version), 0)
            FROM transcriptions
            WHERE hash = ? AND doctor = ?
        ''', (hash_value, doctor))
        current_version = cursor.fetchone()[0] + 1
        
        # Get chunks for this version only
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        chunks = cursor.execute('''
            SELECT transcription 
            FROM chunk_transcriptions 
            WHERE hash = ? AND doctor = ? AND version = ?
            ORDER BY chunk_number
        ''', (hash_value, doctor, current_version)).fetchall()
        conn.close()
        
        # Combine all transcriptions
        full_transcript = " ".join([chunk[0] for chunk in chunks])
        
        # Generate summary using the specified LLM
        summary = generate_llm_summary(full_transcript, prompt, model_name)
        
        # Save the complete transcription and summary
        version = save_transcription(hash_value, doctor, full_transcript, summary, prompt, model_name)
        
        return jsonify({
            "version": version,
            "raw_transcript": full_transcript,
            "summary": summary
        }), 200
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def find_text_overlap(prev_chunk, current_chunk):
    """Find overlapping text between chunks using difflib"""
    from difflib import SequenceMatcher
    
    # Convert to words for better matching
    prev_words = prev_chunk['text'].split()
    curr_words = current_chunk.split()
    
    # Look for matching sequences at end of prev and start of current
    matcher = SequenceMatcher(None, 
                            ' '.join(prev_words[-20:]),  # Last 20 words
                            ' '.join(curr_words[:20]))   # First 20 words
    
    matches = matcher.get_matching_blocks()
    if matches:
        return {
            'text': matcher.a[matches[0].a:matches[0].a + matches[0].size],
            'position_prev': matches[0].a,
            'position_curr': matches[0].b
        }
    return None

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def generate_llm_summary(transcript, prompt, model_name):
    """Generate a summary using the LLM"""
    try:
        full_prompt = prompt + "\nHere is the transcription: " + transcript
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": full_prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            response_json = response.json()
            if "response" in response_json:
                return response_json["response"]
        
        raise Exception(f"Failed to generate summary: {response.status_code}")
    except Exception as e:
        print(f"Error generating summary: {e}")
        raise

def save_transcription(hash_value, doctor, transcript, summary, prompt, model_name):
    """Save the complete transcription and summary to the database"""
    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        # Get next version
        cursor.execute('''
            SELECT COALESCE(MAX(version), 0) + 1
            FROM transcriptions
            WHERE hash = ? AND doctor = ?
        ''', (hash_value, doctor))
        version = cursor.fetchone()[0]
        
        # Insert new transcription
        cursor.execute('''
            INSERT INTO transcriptions 
            (hash, doctor, version, prompt_used, raw_transcript, summary, model_used, audio_saved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (hash_value, doctor, version, prompt, transcript, summary, model_name, False))
        
        conn.commit()
        conn.close()
        return version
    except Exception as e:
        print(f"Error saving transcription: {e}")
        raise

def process_transcription(audio_path, wav_path, chunk_number, hash_value, doctor, version):
    """Process a single transcription request"""
    try:
        logging.debug(f"Processing chunk {chunk_number}")
        
        # Convert to WAV using ffmpeg with detailed error output
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'webm',  # Explicitly specify input format
            '-i', audio_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            wav_path
        ]

        # Run ffmpeg with full error capture
        process = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            logging.error(f"FFmpeg error output for chunk {chunk_number}:")
            logging.error(f"Command: {' '.join(ffmpeg_cmd)}")
            logging.error(f"stderr: {process.stderr}")
            logging.error(f"stdout: {process.stdout}")
            raise subprocess.CalledProcessError(process.returncode, ffmpeg_cmd)

        # Check if the WAV file was created and has content
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise Exception(f"WAV file not created or empty: {wav_path}")

        logging.debug(f"FFmpeg conversion successful for chunk {chunk_number}")

        # Transcribe using Whisper
        result = model.transcribe(wav_path)
        transcription = result["text"].strip()
        logging.debug(f"Transcription complete for chunk {chunk_number}: {transcription}")
        return transcription

    except Exception as e:
        logging.error(f"Error processing chunk {chunk_number}: {str(e)}")
        # Log the full traceback for debugging
        logging.error(traceback.format_exc())
        raise

def clean_transcription(text):
    """Clean and normalize transcription text"""
    if not text:
        return ""
    
    # Remove any non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    # Convert to basic ASCII where possible, but preserve valid Unicode
    try:
        # Try to normalize Unicode characters
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        # Remove any remaining problematic characters
        text = ''.join(char for char in text if ord(char) < 65536)
        
        return text.strip()
    except Exception as e:
        logging.error(f"Error cleaning transcription: {e}")
        # If all else fails, return ASCII-only version
        return text.encode('ascii', 'ignore').decode('ascii').strip()

# Cleanup on shutdown
@atexit.register
def cleanup():
    logging.info("Shutting down server...")
    SHUTDOWN_EVENT.set()
    executor.shutdown(wait=True)
    logging.info("Server shutdown complete")

@app.route('/get_transcription', methods=['GET'])
def get_transcription():
    hash_value = request.args.get('hash')
    doctor = request.args.get('doctor')
    version = request.args.get('version')
    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        

        # Get all transcriptions for this session in order
        cursor.execute('''
            SELECT transcription 
            FROM chunk_transcriptions 
            WHERE hash = ? AND doctor = ? AND version = ?
            ORDER BY chunk_number ASC
        ''', (hash_value, doctor, version))
        
        result = cursor.fetchall()
        
        # Concatenate all transcriptions with spaces between them
        full_transcription = ' '.join(row[0] for row in result if row[0])
        
        # Clean the concatenated transcription
        full_transcription = clean_transcription(full_transcription)

        if result:
            return jsonify({'transcription': result[0]})
        else:
            return jsonify({'transcription': ''})
            
    except Exception as e:
        logging.error(f"Error in get_transcription: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        if 'conn' in locals():
            conn.close()


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=("cert.pem", "key.pem"))
