import sqlite3
from datetime import datetime
from typing import Optional, Dict
import json

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize database tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''CREATE TABLE IF NOT EXISTS encrypted_files
                    (id TEXT PRIMARY KEY,
                     encrypted_data BLOB NOT NULL,
                     salt BLOB NOT NULL,
                     creation_date TIMESTAMP NOT NULL,
                     voice_key TEXT,
                     recognized_text TEXT,
                     features TEXT)''')
                
                conn.commit()
                print("Database initialized successfully")
                
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

    def save_encrypted_file(self, 
                          file_id: str,
                          encrypted_data: bytes,
                          salt: bytes,
                          voice_key: str,
                          recognized_text: str,
                          features: Dict) -> bool:
        """Save encrypted file data to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO encrypted_files
                    (id, encrypted_data, salt, creation_date, voice_key, 
                     recognized_text, features)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (file_id, encrypted_data, salt, datetime.now(), 
                     voice_key, recognized_text, json.dumps(features)))
                return True
        except Exception as e:
            print(f"Error saving encrypted file: {e}")
            return False

    def get_encrypted_file(self, file_id: str) -> Optional[Dict]:
        """Retrieve encrypted file data from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT * FROM encrypted_files 
                                WHERE id = ?''', (file_id,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        'encrypted_data': result['encrypted_data'],
                        'salt': result['salt'],
                        'creation_date': result['creation_date'],
                        'recognized_text': result['recognized_text'],
                        'voice_key': result['voice_key'],
                        'features': json.loads(result['features']) if result['features'] else {}
                    }
                return None
                
        except Exception as e:
            print(f"Error retrieving encrypted file: {e}")
            return None

    def cleanup_old_files(self, days: int = 30):
        """Remove files older than specified days"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''DELETE FROM encrypted_files 
                                WHERE creation_date < datetime('now', '-? days')''',
                             (days,))
                conn.commit()
                print(f"Removed {cursor.rowcount} old files")
        except Exception as e:
            print(f"Error cleaning up old files: {e}")