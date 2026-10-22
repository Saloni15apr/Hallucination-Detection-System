
import sqlite3

DATABASE = "database.db"


def get_connection():
    return sqlite3.connect(DATABASE)


def init_database():

    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Analysis history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            prompt TEXT NOT NULL,
            score REAL NOT NULL,
            label TEXT NOT NULL,
            explanation TEXT
        )
    """)

    conn.commit()
    conn.close()


def register_user(name, email, password):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, password)
        )

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


def login_user(email, password):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE email = ? AND password = ?",
        (email, password)
    )

    user = cursor.fetchone()

    conn.close()

    return user


def save_analysis(email, prompt, score, label, explanation):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history
        (email, prompt, score, label, explanation)
        VALUES (?, ?, ?, ?, ?)
        """,
        (email, prompt, score, label, explanation)
    )

    conn.commit()
    conn.close()


def get_history(email):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT prompt, score, label, explanation
        FROM history
        WHERE email = ?
        ORDER BY id DESC
        """,
        (email,)
    )

    history = cursor.fetchall()

    conn.close()

    return history
