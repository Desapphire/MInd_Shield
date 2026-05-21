"""
Encrypted local SQLite storage helpers for Mind-Shield+.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, Iterable, Optional

from security import LocalCipher


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


class LocalStorage:
    """SQLite storage with encrypted text fields."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(DATA_DIR, "mindshield_local.db")
        self.cipher = LocalCipher()
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    face_profile TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    started_at TEXT,
                    ended_at TEXT,
                    summary TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
                CREATE TABLE IF NOT EXISTS fatigue_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    payload TEXT
                );
                CREATE TABLE IF NOT EXISTS emotion_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    payload TEXT
                );
                CREATE TABLE IF NOT EXISTS posture_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    payload TEXT
                );
                CREATE TABLE IF NOT EXISTS burnout_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    payload TEXT
                );
                CREATE TABLE IF NOT EXISTS adaptive_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    payload TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS focus_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    payload TEXT
                );
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    payload TEXT
                );
                CREATE TABLE IF NOT EXISTS privacy_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    value TEXT
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_payload(self, table: str, payload: Dict[str, Any], ts: Optional[str] = None) -> None:
        conn = self._connect()
        try:
            encrypted = self.cipher.encrypt_json(payload)
            ts_value = ts or payload.get("ts") or payload.get("timestamp") or ""
            conn.execute(f"INSERT INTO {table} (ts, payload) VALUES (?, ?)", (ts_value, encrypted))
            conn.commit()
        finally:
            conn.close()

    def set_setting(self, key: str, value: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            encrypted = self.cipher.encrypt_json(value)
            conn.execute(
                "INSERT INTO privacy_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, encrypted),
            )
            conn.commit()
        finally:
            conn.close()

    def get_setting(self, key: str) -> Dict[str, Any]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT value FROM privacy_settings WHERE key=?", (key,))
            row = cur.fetchone()
            if not row:
                return {}
            return self.cipher.decrypt_json(row[0])
        finally:
            conn.close()
