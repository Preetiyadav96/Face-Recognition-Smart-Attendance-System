
import sqlite3
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "attendance.db"
)

os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)


# ---------------- DATABASE CONNECTION ----------------

def get_connection():
    return sqlite3.connect(DATABASE_PATH, timeout=10)


# ---------------- CREATE TABLE ----------------

def init_db():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        time TEXT,
        status TEXT
    )
    """)

    cursor.execute("PRAGMA table_info(attendance)")
    columns = [column[1] for column in cursor.fetchall()]

    if "employee_id" not in columns:
        cursor.execute(
            "ALTER TABLE attendance ADD COLUMN employee_id TEXT"
        )

    conn.commit()
    conn.close()


# ---------------- MARK ATTENDANCE ----------------

def mark_attendance(employee_id, name):

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    try:

        cursor.execute("""
        SELECT id FROM attendance
        WHERE employee_id = ? AND date = ?
        """, (str(employee_id), date))

        existing = cursor.fetchone()

        if existing:
            return "Already Marked"

        cursor.execute("""
        INSERT INTO attendance
        (employee_id, name, date, time, status)
        VALUES (?, ?, ?, ?, ?)
        """, (
            str(employee_id),
            name,
            date,
            time,
            "Present"
        ))

        conn.commit()

        return "Attendance Marked"

    except sqlite3.Error as e:
        return f"Database Error: {e}"

    finally:
        conn.close()


# ---------------- VIEW ATTENDANCE ----------------

def view_attendance():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM attendance ORDER BY id DESC
    """)

    records = cursor.fetchall()
    conn.close()

    return records


# ---------------- INITIALIZE DATABASE ----------------

init_db()

if __name__ == "__main__":
    print("Attendance Database Ready!")
    print(view_attendance())
