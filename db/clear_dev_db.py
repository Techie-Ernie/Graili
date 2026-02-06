import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.engine import engine


def clear_dev_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE collection_documents RESTART IDENTITY CASCADE;"))
        conn.execute(text("TRUNCATE TABLE collections RESTART IDENTITY CASCADE;"))
        conn.execute(text("TRUNCATE TABLE questions RESTART IDENTITY CASCADE;"))
        conn.execute(text("TRUNCATE TABLE uploaded_questions RESTART IDENTITY CASCADE;"))
        conn.execute(text("TRUNCATE TABLE subtopics RESTART IDENTITY CASCADE;"))


if __name__ == "__main__":
    clear_dev_db()
    print("Database cleared.")
