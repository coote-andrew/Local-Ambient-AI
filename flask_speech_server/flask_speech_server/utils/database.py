import sqlite3
from datetime import datetime
import logging

def init_db():
    try:
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()

        # First, check if we need to migrate the chunk_transcriptions table
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='chunk_transcriptions'
        ''')
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check if version column exists
            cursor.execute('PRAGMA table_info(chunk_transcriptions)')
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'version' not in columns:
                # Backup old data
                cursor.execute('ALTER TABLE chunk_transcriptions RENAME TO chunk_transcriptions_old')
                
                # Create new table with version
                cursor.execute('''
                    CREATE TABLE chunk_transcriptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        hash TEXT NOT NULL,
                        doctor TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        chunk_time INTEGER NOT NULL,
                        transcription TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(hash, doctor, version, chunk_time)
                    )
                ''')
                
                # Copy old data with default version 1, handling duplicates
                cursor.execute('''
                    INSERT INTO chunk_transcriptions 
                    (hash, doctor, version, chunk_time, transcription, created_at)
                    SELECT 
                        hash, 
                        doctor, 
                        1, 
                        chunk_time, 
                        transcription,
                        created_at
                    FROM (
                        SELECT 
                            hash, 
                            doctor, 
                            chunk_time, 
                            transcription,
                            created_at,
                            ROW_NUMBER() OVER (
                                PARTITION BY hash, doctor, chunk_time 
                                ORDER BY created_at DESC
                            ) as rn
                        FROM chunk_transcriptions_old
                    ) ranked
                    WHERE rn = 1
                ''')
                
                # Drop old table
                cursor.execute('DROP TABLE chunk_transcriptions_old')
        else:
            # Create new table if it doesn't exist
            cursor.execute('''
                CREATE TABLE chunk_transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash TEXT NOT NULL,
                    doctor TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    chunk_time INTEGER NOT NULL,
                    transcription TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(hash, doctor, version, chunk_time)
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
                prompt_used TEXT NOT NULL,
                raw_transcript TEXT NOT NULL,
                summary TEXT,
                model_used TEXT,
                audio_saved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT NOT NULL,
                doctor TEXT NOT NULL,
                version INTEGER NOT NULL,
                chunk_time INTEGER NOT NULL,
                chunk_data BLOB NOT NULL
            )
        ''')

        # Insert default LLM if not exists
        cursor.execute('''
            INSERT OR IGNORE INTO llm_models (name, is_active, is_default)
            VALUES ('qwen2.5', 1, 1)
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