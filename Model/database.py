import sqlite3
from datetime import datetime

DB_PATH = 'feedback.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        features TEXT,
        timestamp TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id TEXT PRIMARY KEY,
        transaction_id TEXT,
        fraud_prob REAL,
        confidence REAL,
        prediction_label TEXT,
        predicted_at TEXT,
        reviewed INTEGER DEFAULT 0,
        FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        feedback_id TEXT PRIMARY KEY,
        prediction_id TEXT,
        reviewer_id TEXT,
        corrected_label TEXT,
        confidence INTEGER,
        reason TEXT,
        feedback_time TEXT,
        FOREIGN KEY (prediction_id) REFERENCES predictions (prediction_id)
    )
    ''')

    conn.commit()
    conn.close()

def _execute_query(query, params=(), fetch=False):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('PRAGMA journal_mode=WAL;')  # Enable WAL mode for better concurrency
        cursor = conn.cursor()
        cursor.execute(query, params)
        if fetch:
            result = cursor.fetchall()
        else:
            result = None
        conn.commit()
        return result
    finally:
        conn.close()

def insert_transaction(transaction_id, features_dict):
    query = 'INSERT OR IGNORE INTO transactions (transaction_id, features, timestamp) VALUES (?, ?, ?)'
    _execute_query(query, (transaction_id, str(features_dict), datetime.utcnow().isoformat()))

def insert_prediction(prediction_id, transaction_id, fraud_prob, confidence, prediction_label):
    query = '''INSERT OR IGNORE INTO predictions 
               (prediction_id, transaction_id, fraud_prob, confidence, prediction_label, predicted_at) 
               VALUES (?, ?, ?, ?, ?, ?)'''
    _execute_query(query, (prediction_id, transaction_id, fraud_prob, confidence, prediction_label, datetime.utcnow().isoformat()))

def update_reviewed(prediction_id):
    query = 'UPDATE predictions SET reviewed=1 WHERE prediction_id=?'
    _execute_query(query, (prediction_id,))

def insert_feedback(feedback_id, prediction_id, reviewer_id, corrected_label, confidence, reason):
    query = '''INSERT INTO feedback 
               (feedback_id, prediction_id, reviewer_id, corrected_label, confidence, reason, feedback_time) 
               VALUES (?, ?, ?, ?, ?, ?, ?)'''
    _execute_query(query, (feedback_id, prediction_id, reviewer_id, corrected_label, confidence, reason, datetime.utcnow().isoformat()))
    update_reviewed(prediction_id)

def get_unreviewed_predictions(limit=10):
    query = 'SELECT * FROM predictions WHERE reviewed=0 ORDER BY predicted_at ASC LIMIT ?'
    return _execute_query(query, (limit,), fetch=True)

def count_feedback():
    query = 'SELECT COUNT(*) FROM feedback'
    result = _execute_query(query, fetch=True)
    return result[0][0] if result else 0
