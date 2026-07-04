import sqlite3                  
import os
from datetime import datetime   
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "queries.db")
# Connect to the database file (creates it if it doesn't exist)
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT,
            question    TEXT,
            answer      TEXT,
            timestamp   TEXT
        )
    ''')
    conn.commit()   # Save changes
    conn.close()    # Close the connection

#Saves one question and its answer into the database.
def save_query(report_name: str, question: str, answer: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO query_history (report_name, question, answer, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (
        report_name,
        question,
        answer,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # e.g. "2025-01-15 10:30:00"
    ))
    conn.commit()
    conn.close()

# Returns all saved questions from the database.
def get_all_queries():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, report_name, question, answer, timestamp
        FROM query_history
        ORDER BY id DESC   -- newest first
    ''')
    rows = cursor.fetchall()   # Get all results as a list
    conn.close()
    return rows

def get_top_questions(limit: int = 5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT question, COUNT(*) as count
        FROM query_history
        GROUP BY question
        ORDER BY count DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows