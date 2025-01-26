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
from datetime import datetime, timedelta
import shutil



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
            if transcription["status"] == "success":
                error = transcription["error"]
                transcription = clean_transcription(transcription["transcription"])
                print(f"*****************Transcription: {transcription}")
                print(f"Chunk number: {chunk_number}")
                # Save transcription for debugging
            else:
                error = transcription["error"]
                transcription = ""
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
                "chunk_number": chunk_number,
                "error": error
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
        transcription = request.form.get('transcription')



        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COALESCE(MAX(version), 0)
            FROM transcriptions
            WHERE hash = ? AND doctor = ?
        ''', (hash_value, doctor))
        current_version = cursor.fetchone()[0] + 1


        # if transciption has the same version as the highest chunk_transcription version, then we don't need to save a copy into chunk_transcriptions, otherwise we do
        cursor.execute('''
            SELECT COALESCE(MAX(version), 0)
            FROM chunk_transcriptions
            WHERE hash = ? AND doctor = ?
        ''', (hash_value, doctor))
        highest_chunk_version = cursor.fetchone()[0]

        if current_version == highest_chunk_version:

            chunks = cursor.execute('''
            SELECT transcription 
            FROM chunk_transcriptions 
            WHERE hash = ? AND doctor = ? AND version = ?
            ORDER BY chunk_number
            ''', (hash_value, doctor, current_version)).fetchall()
            full_transcript = " ".join([chunk[0] for chunk in chunks])
        else:
            print("saving chunk_transcriptions")
            cursor.execute('''
                INSERT INTO chunk_transcriptions 
                (hash, doctor, version, chunk_number, transcription)
                VALUES (?, ?, ?, ?, ?)
            ''', (hash_value, doctor, current_version, 0, transcription))
            conn.commit()

            full_transcript = transcription

        
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
        # Get transcription tips from database
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        cursor.execute('SELECT error, fix FROM transcription_tips')
        tips = cursor.fetchall()
        conn.close()

        # Format tips for prompt
        tips_text = "\n\nCommon transcription considerations:\n"
        for error, fix in tips:
            if error:
                tips_text += f"- When encountering {error}: {fix}\n"
            else:
                tips_text += f"- Important: {fix}\n"

        # Combine prompt with transcript and tips
        full_prompt = f"{prompt}\n\nHere is the transcription: {transcript}{tips_text}"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_ctx": 50000,
                }
            }
        )
        
        if response.status_code == 200:
            response_json = response.json()
            if "response" in response_json:
                response = response_json["response"].replace("---\n", "")
                return response
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
        
        # Check if original audio file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logging.debug(f"Empty audio file for chunk {chunk_number}")
            print(f"Empty audio file for chunk {chunk_number}")
            return {"status": "success", "transcription": "", "error": "Empty chunk"}

        # Check audio levels using ffmpeg
        silence_check_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 'webm',  # Force input format
            '-i', audio_path,
            '-af', 'volumedetect',
            '-f', 'null',
            '-'
        ]
        
        silence_process = subprocess.run(
            silence_check_cmd,
            capture_output=True,
            text=True
        )
        
        # Parse the mean volume from ffmpeg output
        mean_volume = None
        for line in silence_process.stderr.split('\n'):
            if 'mean_volume' in line:
                try:
                    mean_volume = float(line.split(':')[1].strip().replace(' dB', ''))
                except (ValueError, IndexError):
                    pass
        
        # If mean volume is very low or couldn't be detected, return empty transcription
        if mean_volume is None or mean_volume < -60:
            logging.debug(f"Silent chunk {chunk_number} detected (mean volume: {mean_volume}dB)")
            print(f"Silent chunk {chunk_number} detected (mean volume: {mean_volume}dB)")
            return {"status": "success", "transcription": "", "error": "Silent chunk"}

        # Convert to WAV using ffmpeg with all standard parameters
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 'webm',  # Force input format
            '-i', audio_path,
            '-acodec', 'pcm_s16le',  # Output codec
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono audio
            wav_path
        ]

        process = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            logging.error(f"FFmpeg error output for chunk {chunk_number}:")
            logging.error(f"Command: {' '.join(ffmpeg_cmd)}")
            logging.error(f"stderr: {process.stderr}")
            return {"status": "success", "transcription": "", "error": ""}

        try:
            # Process audio with Whisper
            result = model.transcribe(wav_path)
            transcription = clean_transcription(result["text"].strip())
            return {"status": "success", "transcription": transcription, "error": ""}
        except RuntimeError as e:
            if "cannot reshape tensor of 0 elements" in str(e):
                # This is likely a silent or no-speech chunk
                return {"status": "success", "transcription": "", "error": ""}
            else:
                # This is an unexpected error, log it
                raise e

    except Exception as e:
        logging.error(f"Error processing chunk {chunk_number}: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Error processing chunk {chunk_number}: {str(e)}")
        print(traceback.format_exc())

        return {"status": "success", "transcription": "", "error": str(e)}
    
    finally:
        # Cleanup temporary files
        if 'wav_path' in locals():
            try:
                os.unlink(wav_path)
            except:
                pass


def clean_transcription(text):
    """Clean and normalize transcription text"""
    if not text:
        return ""
    
    # Always convert to ASCII for database storage
    try:
        # First try to normalize Unicode characters
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        # Then force to ASCII
        return text.encode('ascii', 'ignore').decode('ascii').strip()
    except Exception as e:
        logging.error(f"Error cleaning transcription: {e}")
        # If all else fails, return empty string
        return ""

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
        sql = '''
            SELECT transcription 
            FROM chunk_transcriptions 
            WHERE hash = ? AND doctor = ? AND version = ?
            ORDER BY chunk_number ASC
        '''
        print(f"Executing SQL: {sql} with params: {(hash_value, doctor, version)}")
        cursor.execute(sql, (hash_value, doctor, version))
        
        result = cursor.fetchall()
        
        # Concatenate all transcriptions with spaces between them
        full_transcription = ' '.join(row[0] for row in result if row[0])

        print("full_transcription: ", full_transcription)
        # Clean the concatenated transcription
        full_transcription = clean_transcription(full_transcription)

        if result:
            return jsonify({'transcription': full_transcription})
            
    except Exception as e:
        logging.error(f"Error in get_transcription: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        if 'conn' in locals():
            conn.close()

@app.route('/download_audio/<hash_value>/<doctor>/<version>', methods=['GET'])
def download_audio(hash_value, doctor, version):
    try:
        # Check if permanent audio exists first
        permanent_path = os.path.join(os.path.abspath('audio_files'), hash_value, doctor, f'recording_v{version}.webm')
        if os.path.exists(permanent_path):
            return send_file(permanent_path, as_attachment=True)

        # If not, concatenate from debug directory
        debug_dir = os.path.join(os.path.abspath('debug_audio'), hash_value, doctor, version)
        if not os.path.exists(debug_dir):
            return jsonify({"error": "Audio not found"}), 404

        # Create temp directory within our debug_audio folder
        temp_dir = os.path.join(debug_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create paths for our temporary files
        temp_path = os.path.join(temp_dir, 'concatenated.webm')
        file_list = os.path.join(temp_dir, 'files.txt')

        # Get all chunk files and sort them
        chunk_files = []
        for chunk_dir in os.listdir(debug_dir):
            if chunk_dir != 'temp':  # Skip our temp directory
                audio_path = os.path.join(debug_dir, chunk_dir, 'original.webm')
                if os.path.exists(audio_path):
                    try:
                        chunk_num = int(chunk_dir)
                        chunk_files.append((chunk_num, os.path.abspath(audio_path)))
                    except ValueError:
                        continue
        
        if not chunk_files:
            return jsonify({"error": "No audio chunks found"}), 404
            
        chunk_files.sort()  # Sort by chunk number

        # Create file list for ffmpeg
        with open(file_list, 'w') as f:
            for _, path in chunk_files:
                # Escape backslashes for ffmpeg
                escaped_path = path.replace('\\', '/')
                f.write(f"file '{escaped_path}'\n")

        try:
            # Concatenate files using ffmpeg
            result = subprocess.run([
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', file_list, '-c', 'copy', temp_path
            ], capture_output=True, text=True, check=True)
            
            return send_file(temp_path, as_attachment=True)

        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error: {e.stderr}")
            return jsonify({"error": f"FFmpeg error: {e.stderr}"}), 500

    except Exception as e:
        logging.error(f"Error downloading audio: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary files
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.error(f"Error cleaning up temp directory: {e}")

@app.route('/save_audio_permanent', methods=['POST'])
def save_audio_permanent():
    hash_value = request.form.get('hash')
    doctor = request.form.get('doctor')
    version = request.form.get('version')

    try:
        # Check if already saved
        permanent_path = os.path.join(os.path.abspath('audio_files'), hash_value, doctor, f'recording_v{version}.webm')
        permanent_dir = os.path.dirname(permanent_path)
        
        if os.path.exists(permanent_path):
            return jsonify({"message": "Audio already saved permanently"}), 200

        # Create permanent directory
        os.makedirs(permanent_dir, exist_ok=True)

        # Get the debug audio directory path
        debug_dir = os.path.join(os.path.abspath('debug_audio'), hash_value, doctor, str(version))
        
        if not os.path.exists(debug_dir):
            raise FileNotFoundError(f"Debug directory not found: {debug_dir}")

        # Create temporary directory for file list
        temp_dir = os.path.join(debug_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Similar concatenation logic as download_audio
        chunk_files = []
        for chunk_dir in os.listdir(debug_dir):
            if chunk_dir != 'temp' and chunk_dir.isdigit():  # Skip temp dir and non-numeric dirs
                audio_path = os.path.join(debug_dir, chunk_dir, 'original.webm')
                if os.path.exists(audio_path):
                    chunk_files.append((int(chunk_dir), audio_path))
        
        if not chunk_files:
            raise FileNotFoundError("No audio chunks found")
            
        chunk_files.sort()  # Sort by chunk number

        # Create file list with absolute paths
        file_list_path = os.path.join(temp_dir, 'files.txt')
        with open(file_list_path, 'w') as f:
            for _, path in chunk_files:
                # Use absolute paths and escape backslashes
                abs_path = os.path.abspath(path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

        # Run ffmpeg with absolute paths
        subprocess.run([
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', file_list_path, 
            '-c', 'copy', 
            permanent_path
        ], check=True)

        # Clean up
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

        # Update database
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE transcriptions 
            SET audio_saved = TRUE
            WHERE hash = ? AND doctor = ? AND version = ?
        ''', (hash_value, doctor, version))
        conn.commit()
        conn.close()

        return jsonify({"message": "Audio saved permanently"}), 200

    except Exception as e:
        logging.error(f"Error saving audio permanently: {e}")
        return jsonify({"error": str(e)}), 500

def cleanup_old_debug_audio():
    """Delete debug audio files older than 48 hours"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=48)
        debug_root = 'debug_audio'

        for hash_dir in os.listdir(debug_root):
            hash_path = os.path.join(debug_root, hash_dir)
            for doctor_dir in os.listdir(hash_path):
                doctor_path = os.path.join(hash_path, doctor_dir)
                for version_dir in os.listdir(doctor_path):
                    version_path = os.path.join(doctor_path, version_dir)
                    dir_time = datetime.fromtimestamp(os.path.getctime(version_path))
                    
                    if dir_time < cutoff_time:
                        shutil.rmtree(version_path)
                        logging.info(f"Cleaned up old debug audio: {version_path}")

                # Clean up empty directories
                if not os.listdir(doctor_path):
                    os.rmdir(doctor_path)
                if not os.listdir(hash_path):
                    os.rmdir(hash_path)

    except Exception as e:
        logging.error(f"Error cleaning up debug audio: {e}")

# Add cleanup scheduling (add this near the end of the file)
def schedule_cleanup():
    """Run cleanup every hour"""
    while not SHUTDOWN_EVENT.is_set():
        cleanup_old_debug_audio()
        time.sleep(3600)  # Sleep for 1 hour

# Start cleanup thread (add this near other thread starts)
cleanup_thread = Thread(target=schedule_cleanup, daemon=True)
cleanup_thread.start()

@app.template_filter('exists_dir')
def exists_dir(path):
    """Check if directory exists and contains audio files"""
    try:
        # Check if directory exists and has any chunk subdirectories
        return os.path.isdir(path) and any(
            os.path.exists(os.path.join(path, chunk_dir, 'original.webm'))
            for chunk_dir in os.listdir(path)
        )
    except Exception:
        return False

@app.route('/check_chunks_processed', methods=['GET'])
def check_chunks_processed():
    hash_value = request.args.get('hash')
    doctor = request.args.get('doctor')
    version = request.args.get('version')
    final_chunk = request.args.get('final_chunk')
    
    if not all([hash_value, doctor, version, final_chunk]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Check if we have processed all chunks up to final_chunk
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        # Query to check if we have all chunks from 0 to final_chunk
        cursor.execute('''
            SELECT COUNT(DISTINCT chunk_number) 
            FROM chunk_transcriptions 
            WHERE hash = ? AND doctor = ? AND version = ? AND chunk_number <= ?
        ''', (hash_value, doctor, version, final_chunk))
        
        processed_count = cursor.fetchone()[0]
        all_processed = processed_count == int(final_chunk) + 1  # +1 because we count from 0
        
        conn.close()
        
        return jsonify({"all_processed": all_processed}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/record_processing_time', methods=['POST'])
def record_processing_time():
    hash_value = request.form.get('hash')
    doctor = request.form.get('doctor')
    version = request.form.get('version')
    processing_time = request.form.get('processing_time')  # in milliseconds
    
    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        # Add processing_time column if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_times
            (hash TEXT, doctor TEXT, version INTEGER, processing_time REAL, 
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
        ''')
        
        cursor.execute('''
            INSERT INTO processing_times (hash, doctor, version, processing_time)
            VALUES (?, ?, ?, ?)
        ''', (hash_value, doctor, version, processing_time))
        
        # Calculate rolling average (last 50 recordings)
        cursor.execute('''
            SELECT AVG(processing_time) FROM (
                SELECT processing_time 
                FROM processing_times 
                ORDER BY timestamp DESC 
                LIMIT 50
            )
        ''')
        
        avg_time = cursor.fetchone()[0] or 10000  # Default to 10 seconds if no data
        
        conn.commit()
        conn.close()
        
        return jsonify({"average_time": avg_time}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_average_processing_time', methods=['GET'])
def get_average_processing_time():
    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT AVG(processing_time) FROM (
                SELECT processing_time 
                FROM processing_times 
                ORDER BY timestamp DESC 
                LIMIT 50
            )
        ''')
        
        avg_time = cursor.fetchone()[0] or 10000  # Default to 10 seconds if no data
        conn.close()
        
        return jsonify({"average_time": avg_time}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_funny_quotes', methods=['GET'])
def get_funny_quotes():
    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        cursor.execute('SELECT quote FROM funny_quotes')
        quotes = [row[0] for row in cursor.fetchall()]
        conn.close()
        return jsonify({"quotes": quotes}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_from_data', methods=['POST'])
def generate_from_data():
    try:
        data = request.get_json()
        if not data or not all(key in data for key in ['prompt', 'model', 'patient_data']):
            return jsonify({
                "status": "error",
                "generated_text": "Missing required fields"
            }), 400

        prompt = data['prompt']
        model = data['model']
        patient_data = data['patient_data']
        
        # Get generation tips from database
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        cursor.execute('SELECT tip FROM data_generation_tips WHERE is_active = TRUE')
        tips = cursor.fetchall()
        
        # Format tips for prompt
        tips_text = "\n\nImportant considerations:\n"
        for (tip,) in tips:
            tips_text += f"- {tip}\n"

        # Combine prompt with patient data and tips
        full_prompt = f"{prompt}\n\nPatient Data:\n{patient_data}{tips_text}"
        
        start_time = time.time()
        
        # Generate response using the specified LLM
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False
            }
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            response_json = response.json()
            if "response" in response_json:
                generated_text = response_json["response"]
                
                # Log the generation
                cursor.execute('''
                    INSERT INTO data_generations 
                    (prompt, patient_data, model_used, generated_text, processing_time)
                    VALUES (?, ?, ?, ?, ?)
                ''', (prompt, patient_data, model, generated_text, processing_time))
                
                conn.commit()
                conn.close()
                
                return jsonify({
                    "status": "success",
                    "generated_text": generated_text
                }), 200
        
        error_msg = f"Failed to generate response: {response.status_code}"
        # Log the error
        cursor.execute('''
            INSERT INTO data_generations 
            (prompt, patient_data, model_used, generated_text, processing_time, error)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (prompt, patient_data, model, "", processing_time, error_msg))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "error",
            "generated_text": error_msg
        }), response.status_code

    except Exception as e:
        logging.error(f"Error in generate_from_data: {str(e)}")
        return jsonify({
            "status": "error",
            "generated_text": str(e)
        }), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=("cert.pem", "key.pem"))
