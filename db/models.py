from sqlalchemy import (
    Column, Integer, Text, String, CheckConstraint,
    ForeignKey,
    TIMESTAMP, UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True)

    question_type = Column(
        String,
        nullable=False
    )  # 'exam' or 'understanding'

    category = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    chapter = Column(String, nullable=False)

    question_text = Column(Text, nullable=False)
    marks = Column(Integer)
    document_name = Column(Text)
    source_link = Column(Text)
    answer_link = Column(Text)

    created_at = Column(TIMESTAMP, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "question_type IN ('exam', 'understanding')",
            name="question_type_check"
        ),
        CheckConstraint(
            """
            (question_type = 'exam' AND marks IS NOT NULL)
            OR
            (question_type = 'understanding' AND marks IS NULL)
            """,
            name="marks_consistency_check"
        ),
    )


class UploadedQuestion(Base):
    __tablename__ = "uploaded_questions"

    id = Column(Integer, primary_key=True)

    question_type = Column(
        String,
        nullable=False
    )  # 'exam' or 'understanding'

    category = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    chapter = Column(String, nullable=False)

    question_text = Column(Text, nullable=False)
    marks = Column(Integer)
    document_name = Column(Text)
    source_link = Column(Text)
    answer_link = Column(Text)

    created_at = Column(TIMESTAMP, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "question_type IN ('exam', 'understanding')",
            name="uploaded_question_type_check"
        ),
        CheckConstraint(
            """
            (question_type = 'exam' AND marks IS NOT NULL)
            OR
            (question_type = 'understanding' AND marks IS NULL)
            """,
            name="uploaded_marks_consistency_check"
        ),
    )

class Subtopic(Base):
    __tablename__ = "subtopics"

    id = Column(Integer, primary_key=True)
    subject = Column(String, nullable=False)
    code = Column(String, nullable=False)
    title = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "subject",
            "code",
            name="subtopic_subject_code_unique"
        ),
    )


class Collection(Base):
    __tablename__ = "collections"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    subject = Column(String, nullable=False)

    created_at = Column(TIMESTAMP, server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "name",
            "subject",
            name="collection_name_subject_unique"
        ),
    )


class CollectionDocument(Base):
    __tablename__ = "collection_documents"

    id = Column(Integer, primary_key=True)
    collection_id = Column(Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    subject = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    document_name = Column(Text, nullable=False)

    created_at = Column(TIMESTAMP, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "source_type IN ('scraped', 'uploaded')",
            name="collection_document_source_type_check"
        ),
        UniqueConstraint(
            "collection_id",
            "subject",
            "source_type",
            "document_name",
            name="collection_document_unique"
        ),
    )
