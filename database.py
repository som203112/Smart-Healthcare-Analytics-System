import sqlite3

# =========================
# CONNECT DATABASE
# =========================

conn = sqlite3.connect(
    "healthcare.db"
)

cursor = conn.cursor()

# =========================
# CREATE PATIENT TABLE
# =========================

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS patients (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        age INTEGER,

        gender TEXT,

        heart_rate INTEGER,

        blood_pressure INTEGER,

        temperature REAL,

        predicted_disease TEXT
    )
    """
)

# =========================
# SAVE CHANGES
# =========================

conn.commit()

print("Database created successfully!")
