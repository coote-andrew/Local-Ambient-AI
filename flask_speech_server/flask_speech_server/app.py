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

    return render_template('audio_record.html', 
                         hash_value=hash_value,
                         doctor=doctor,
                         llm_models=llm_models,
                         prompts=prompts,
                         previous_recordings=previous_recordings)

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
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    hash_value = request.form.get('hash')
    doctor = request.form.get('doctor')
    chunk_time = request.form.get('chunk_time', 0)
    is_final_chunk = request.form.get('is_final_chunk', 'false').lower() == 'true'
    
    try:
        # Save the raw PCM data
        temp_audio_path = tempfile.mktemp(suffix='.raw')
        wav_file_path = tempfile.mktemp(suffix='.wav')
        
        audio_file.save(temp_audio_path)
        
        # Convert raw PCM to WAV
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 's16le',  # 16-bit signed little-endian
            '-ar', '16000',  # Sample rate
            '-ac', '1',      # Mono
            '-i', temp_audio_path,
            wav_file_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Process the audio...
        result = model.transcribe(wav_file_path)
        transcription = result["text"]
        
        # Store in database
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COALESCE(MAX(version), 0)
            FROM chunk_transcriptions
            WHERE hash = ? AND doctor = ?
        ''', (hash_value, doctor))
        current_version = cursor.fetchone()[0]
        version = current_version if current_version > 0 else get_next_version(hash_value, doctor)
        
        cursor.execute('''
            INSERT OR REPLACE INTO chunk_transcriptions 
            (hash, doctor, version, chunk_time, transcription)
            VALUES (?, ?, ?, ?, ?)
        ''', (hash_value, doctor, version, chunk_time, transcription))
            
        conn.commit()
        conn.close()
        
        return jsonify({
            "transcription": transcription,
            "chunk_time": chunk_time,
            "is_final": is_final_chunk,
            "version": version
        }), 200
        
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Cleanup
        for path in [temp_audio_path, wav_file_path]:
            if path and os.path.exists(path):
                os.remove(path)

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
            ORDER BY chunk_time
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

def process_transcription(audio_path, wav_path, hash_value, doctor, chunk_time, is_final_chunk, version, temp_paths=None):
    """Process a single transcription chunk"""
    try:
        if is_final_chunk:
            time.sleep(0.5)
        
        # FFmpeg conversion
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'webm',
            '-i', audio_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            wav_path
        ]
        
        try:
            process = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0 and not os.path.exists(wav_path):
                print("FFmpeg conversion failed:")
                print("stderr:", process.stderr)
                raise Exception(f"FFmpeg conversion failed with return code {process.returncode}")
            
            # Transcribe
            result = model.transcribe(wav_path)
            transcription = result["text"]
            
            # Store in database with REPLACE to handle duplicates
            conn = sqlite3.connect('app.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO chunk_transcriptions 
                (hash, doctor, version, chunk_time, transcription)
                VALUES (?, ?, ?, ?, ?)
            ''', (hash_value, doctor, version, chunk_time, transcription))
            conn.commit()
            conn.close()
            
            return transcription
            
        except subprocess.CalledProcessError as e:
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                # If we have a valid WAV file, continue despite FFmpeg warnings
                pass
            else:
                raise
            
    finally:
        # Clean up temporary files
        if temp_paths:
            for path in temp_paths:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception as e:
                        if os.path.exists(path):
                            print(f"Error cleaning up file {path}: {e}")

# Cleanup on shutdown
@atexit.register
def cleanup():
    SHUTDOWN_EVENT.set()
    executor.shutdown(wait=True)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=("cert.pem", "key.pem"))
