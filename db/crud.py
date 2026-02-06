from sqlalchemy import text
from db.engine import engine
from db.models import Base

def create_schema():
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS idx_questions_subject_year;"))
        conn.execute(text("DROP INDEX IF EXISTS idx_uploaded_questions_subject_year;"))
        conn.execute(text("ALTER TABLE questions DROP COLUMN IF EXISTS year;"))
        conn.execute(text("ALTER TABLE uploaded_questions DROP COLUMN IF EXISTS year;"))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_questions_type
            ON questions (question_type);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_questions_subject
            ON questions (subject);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_questions_chapter
            ON questions (chapter);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_questions_marks
            ON questions (marks);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_uploaded_questions_type
            ON uploaded_questions (question_type);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_uploaded_questions_subject
            ON uploaded_questions (subject);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_uploaded_questions_chapter
            ON uploaded_questions (chapter);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_uploaded_questions_marks
            ON uploaded_questions (marks);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_subtopics_subject
            ON subtopics (subject);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_subtopics_code
            ON subtopics (code);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_collections_subject
            ON collections (subject);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_collections_name
            ON collections (name);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_collection_documents_collection
            ON collection_documents (collection_id);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_collection_documents_subject_source_doc
            ON collection_documents (subject, source_type, document_name);
        """))
