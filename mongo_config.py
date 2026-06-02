from pymongo import MongoClient
import gridfs
import datetime
import pytz

# --- CONFIG ---
# Standard Localhost URI (MongoDB is running on your server)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "JindalAI" 
IST = pytz.timezone('Asia/Kolkata')

class MongoDBHandler:
    def __init__(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            # Stores conversation audio
            self.fs = gridfs.GridFS(self.db) 
            # Stores text logs
            self.logs = self.db["interaction_logs"]
            # This is where the Book text lives
            self.book_collection = self.db["biography_content"]
            print(f"? MongoDB Connected: {DB_NAME}")
        except Exception as e:
            print(f"? MongoDB Error: {e}")

    def save_log(self, user_text, ai_text, audio_bytes=None):
        file_id = self.fs.put(audio_bytes, filename="reply.mp3") if audio_bytes else None
        self.logs.insert_one({
            "timestamp": datetime.datetime.now(IST),
            "user": user_text,
            "ai": ai_text,
            "audio_id": file_id
        })

# Global instance
db = MongoDBHandler()