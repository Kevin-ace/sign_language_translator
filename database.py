import sqlite3

def init_db():
    conn = sqlite3.connect('sign_language.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sign_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gesture TEXT NOT NULL,
            text TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_sign_mapping(gesture, text):
    conn = sqlite3.connect('sign_language.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO sign_mappings (gesture, text) VALUES (?, ?)', (gesture, text))
    conn.commit()
    conn.close()

def add_feedback(feedback):
    conn = sqlite3.connect('sign_language.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO feedback (feedback) VALUES (?)', (feedback,))
    conn.commit()
    conn.close()