
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "analysis_history.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at TEXT NOT NULL
                    DEFAULT (datetime('now', 'localtime')),
                file_name TEXT NOT NULL,
                rpm INTEGER,
                run_number TEXT,
                prediction TEXT NOT NULL,
                mse REAL NOT NULL,
                threshold REAL NOT NULL,
                low_z REAL,
                mid_z REAL,
                high_z REAL
            )
        """)


def save_analysis(
    file_name,
    rpm,
    run_number,
    prediction,
    mse,
    threshold,
    low_z=None,
    mid_z=None,
    high_z=None,
):
    initialize_database()

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO analysis_history (
                file_name, rpm, run_number,
                prediction, mse, threshold,
                low_z, mid_z, high_z
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_name,
            rpm,
            run_number,
            prediction,
            mse,
            threshold,
            low_z,
            mid_z,
            high_z,
        ))

        return cursor.lastrowid


def load_history():
    initialize_database()

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT *
            FROM analysis_history
            ORDER BY id DESC
        """).fetchall()

    return [dict(row) for row in rows]