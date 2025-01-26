import sqlite3
from datetime import datetime
import logging

def init_db():
    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT NOT NULL,
                doctor TEXT NOT NULL,
                version INTEGER NOT NULL,
                chunk_number TEXT NOT NULL,
                transcription TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (hash, doctor, version) 
                    REFERENCES transcriptions(hash, doctor, version)
            )
        ''')

        # Create other tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                is_default BOOLEAN DEFAULT FALSE,
                priority INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT NOT NULL,
                doctor TEXT NOT NULL,
                version INTEGER NOT NULL,
                prompt_used TEXT,
                raw_transcript TEXT,
                summary TEXT,
                model_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                audio_saved BOOLEAN DEFAULT FALSE,
                UNIQUE(hash, doctor, version)
            )
        ''')

        # Insert default LLM if not exists
        cursor.execute('''
            INSERT OR IGNORE INTO llm_models (name, is_active, is_default)
            VALUES ('qwen2.5:7b', 1, 1)
        ''')

        # Insert global default prompt if no default exists for ~All
        cursor.execute('''
            INSERT OR IGNORE INTO prompts (doctor_id, prompt_text, is_default, priority)
            SELECT '~All', 
                   'You are an AI assistant at an ophthalmology clinic in Australia. You are given a transcription of a user''s voice with a patient talking as well. Your task is to provide a summarised account of the visit. Here is the transcription:', 
                   1, 
                   0
            WHERE NOT EXISTS (
                SELECT 1 FROM prompts 
                WHERE doctor_id = '~All' 
                AND is_default = 1
            )
            AND NOT EXISTS (
                SELECT 1 FROM prompts 
                WHERE doctor_id = '~All'
            )
        ''')

            # Add processing_times table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_times
            (hash TEXT, 
            doctor TEXT, 
            version INTEGER, 
            processing_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (hash, doctor, version) REFERENCES transcriptions(hash, doctor, version))
        ''')

        # Create index for faster average calculations
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_processing_times_timestamp 
            ON processing_times(timestamp DESC)
        ''')
        # Create funny_quotes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS funny_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quote TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert default funny quotes if table is empty
        cursor.execute('SELECT COUNT(*) FROM funny_quotes')
        if cursor.fetchone()[0] == 0:
            quotes = [
                "Teaching AI to understand doctor handwriting...",
                "Consulting with virtual medical board...",
                "Debugging the human condition...",
                "Optimizing healthcare one recording at a time...",
                "Translating medical jargon to English...",
                "Calculating the meaning of life (and medicine)...",
                "Performing digital surgery on your audio...",
                "Consulting with Dr. ChatGPT...",
                "Applying machine learning bandages...",
                "Prescribing ones and zeros..."
            ]
            cursor.executemany('INSERT INTO funny_quotes (quote) VALUES (?)', 
                            [(quote,) for quote in quotes])

        # Add transcription_tips table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcription_tips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error TEXT,
                fix TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert default transcription tips if table is empty
        cursor.execute('SELECT COUNT(*) FROM transcription_tips')
        if cursor.fetchone()[0] == 0:
            tips = [
                (None, "Include any follow-up appointments or recommendations"),
                (None, "List all medications discussed during the consultation"),
                (None, "List all medical conditions discussed during the consultation, remember - these are likely to be ophthalmology related"),
                (None, "Do not provide tips if the transcription is faulty, just provide a summary of the consultation"),
                (None, "You do not need to provide date, patient name, or doctor name in the summary - this information will be pasted into a note section by the doctor. The other information is already available.")
            ]
            cursor.executemany('INSERT INTO transcription_tips (error, fix) VALUES (?, ?)', tips)
            conn.commit()

        # Create data_generations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                patient_data TEXT NOT NULL,
                model_used TEXT NOT NULL,
                generated_text TEXT NOT NULL,
                processing_time REAL,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create data_generation_tips table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_generation_tips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tip TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create index for faster querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_data_generations_created_at 
            ON data_generations(created_at DESC)
        ''')

        conn.commit()
        conn.close()
        logging.info("Database initialized successfully")
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        print(f"Error initializing database: {e}")
        raise

def get_next_version(hash_value, doctor):
    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COALESCE(MAX(version), 0) + 1
            FROM transcriptions
            WHERE hash = ? AND doctor = ?
        ''', (hash_value, doctor))
        
        next_version = cursor.fetchone()[0]
        conn.close()
        return next_version
    except sqlite3.Error as e:
        logging.error(f"Error getting next version: {e}")
        print(f"Error getting next version: {e}")
        raise