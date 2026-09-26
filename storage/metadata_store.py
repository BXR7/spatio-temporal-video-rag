"""SQLite exact metadata store for ASR words and spatial OCR regions."""
import os
import sqlite3
import json


class TemporalMetadataStore:
    def __init__(self, db_path="storage/temporal_metadata.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS word_timestamps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                word TEXT NOT NULL,
                start_ts REAL NOT NULL,
                end_ts REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                frame_id TEXT,
                timestamp REAL NOT NULL,
                text TEXT NOT NULL,
                confidence REAL,
                bbox_json TEXT,
                bbox_norm_json TEXT,
                frame_path TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_word_video_clip ON word_timestamps(video_id, clip_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocr_video_clip ON ocr_regions(video_id, clip_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocr_timestamp ON ocr_regions(timestamp)")
        self.conn.commit()

    def store_word_timestamps(self, clip_id, video_id, words):
        rows = [
            (clip_id, video_id, str(word["word"]), float(word["start"]), float(word["end"]))
            for word in words if {"word", "start", "end"} <= set(word)
        ]
        if rows:
            self.conn.executemany(
                "INSERT INTO word_timestamps (clip_id, video_id, word, start_ts, end_ts) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()

    def store_ocr_observations(self, clip_id, video_id, observations):
        rows = []
        for observation in observations:
            for region in observation.get("regions", []):
                rows.append((
                    clip_id,
                    video_id,
                    observation.get("frame_id", ""),
                    float(observation.get("timestamp", 0.0)),
                    region.get("text", ""),
                    float(region.get("confidence", 0.0)),
                    json.dumps(region.get("bbox", [])),
                    json.dumps(region.get("bbox_norm", [])),
                    observation.get("path", ""),
                ))
        if rows:
            self.conn.executemany(
                """INSERT INTO ocr_regions
                (clip_id, video_id, frame_id, timestamp, text, confidence, bbox_json, bbox_norm_json, frame_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self.conn.commit()

    def clear_video(self, video_id):
        self.conn.execute("DELETE FROM word_timestamps WHERE video_id=?", (video_id,))
        self.conn.execute("DELETE FROM ocr_regions WHERE video_id=?", (video_id,))
        self.conn.commit()
