from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pymupdf
from backend.scraper import HolyGrailScraper
import glob 
import asyncio
import re
from difflib import SequenceMatcher
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, or_, select
from db.engine import engine
from db.models import Collection, CollectionDocument, Question, Subtopic, UploadedQuestion
from db.crud import create_schema
from backend.syllabus import extract_clean_body_text, save_syllabus_text
from pathlib import Path
from typing import Any, Dict, Optional, Literal
from threading import Lock
import os

app = FastAPI()

def _parse_cors_origins() -> list[str]:
    raw = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,https://graili-app.onrender.com",
    )
    origins = [value.strip() for value in raw.split(",") if value.strip()]
    return origins or ["http://localhost:5173", "https://graili-app.onrender.com"]

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScrapedData(BaseModel):
    text: str

class AIResult(BaseModel):
    result: str

class QuestionContext(BaseModel):
    year: int
    subject: str
    category: str
    question_type: str
    source_link: str
    answer_link: Optional[str] = None
    document_name: str
    source_type: Literal["scraped", "uploaded"] = "scraped"
    client_session_id: Optional[str] = None

class AIResultRequest(BaseModel):
    result: Dict[str, Any]
    context: Optional[QuestionContext] = None

class SubtopicCreate(BaseModel):
    subject: str
    code: str
    title: str

class SubtopicBulkCreate(BaseModel):
    subject: str
    subtopics: list[Dict[str, str]]

class CollectionCreate(BaseModel):
    name: str
    subject: str

class CollectionDocumentCreate(BaseModel):
    collection_id: int
    subject: str
    source_type: Literal["scraped", "uploaded"]
    document_name: str

class ScraperConfig(BaseModel):
    category: str = "GCE 'A' Levels"
    subject: str = "H2 Economics"
    year: Optional[int] = None
    document_type: str = "Exam Papers"
    pages: int = 3
    subject_label: Optional[str] = None

_context_lock = Lock()
_current_context: Optional[QuestionContext] = None
_scraper_config_lock = Lock()
_scraper_config = ScraperConfig()
_document_cache_lock = Lock()
_document_cache: dict[str, dict[str, Any]] = {}
_document_cache_key: Optional[tuple] = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_ROOT = Path(os.environ.get("DOCUMENTS_ROOT", str(PROJECT_ROOT / "documents")))
SYLLABI_DIR = Path(os.environ.get("SYLLABI_DIR", str(PROJECT_ROOT / "syllabi")))
TMP_DIR = Path(os.environ.get("TMP_DIR", str(PROJECT_ROOT / "tmp")))

@app.on_event("startup")
def initialize_schema() -> None:
    create_schema()

_SUBJECT_CODE_PAREN_RE = re.compile(
    r"^(?P<name>.*?)\s*\(\s*(?:Syllabus\s*)?(?P<code>\d{4})\s*\)\s*$",
    flags=re.IGNORECASE,
)


def _lookup_canonical_subject_by_base(subject_base: str) -> Optional[str]:
    base = " ".join(str(subject_base).split()).strip()
    if not base:
        return None

    # Resolve "Economics" -> "Economics (Syllabus 9750)" (or similar) if we can find a unique match.
    # We intentionally avoid hardcoded maps and derive from the subtopics table.
    with Session(engine) as session:
        rows = session.execute(select(Subtopic.subject).distinct()).all()

    candidates: set[str] = set()
    for (value,) in rows:
        if not value:
            continue
        cleaned = " ".join(str(value).split()).strip()
        cleaned_base = cleaned.split(" (", 1)[0].strip()
        if cleaned_base.lower() == base.lower():
            candidates.add(cleaned)

    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def normalize_subject_label(subject: Optional[str]) -> str:
    if subject is None:
        return ""
    cleaned = " ".join(str(subject).split()).strip()
    if not cleaned:
        return ""

    # Standardize "(9750)" / "(Syllabus9750)" / "(Syllabus 9750)" -> "(Syllabus 9750)".
    match = _SUBJECT_CODE_PAREN_RE.match(cleaned)
    if match:
        name = match.group("name").strip()
        code = match.group("code").strip()
        if name and code:
            return f"{name} (Syllabus {code})"

    cleaned = re.sub(
        r"\(\s*Syllabus\s*(\d{4})\s*\)",
        r"(Syllabus \1)",
        cleaned,
        flags=re.IGNORECASE,
    )

    resolved = _lookup_canonical_subject_by_base(cleaned)
    return resolved or cleaned

def derive_scraper_subject(subject: Optional[str]) -> str:
    cleaned = normalize_subject_label(subject)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return ""

    # If user already passed a Grail-format subject (e.g. "H2 Economics"), normalize it.
    if re.match(r"^H[123]\s+", cleaned, flags=re.IGNORECASE):
        parts = cleaned.split(None, 1)
        if len(parts) == 1:
            return parts[0].upper()
        base = parts[1].split(" (", 1)[0].strip()
        return f"{parts[0].upper()} {base}" if base else parts[0].upper()

    base = cleaned.split(" (", 1)[0].strip()
    if not base:
        return cleaned

    prefix = (os.environ.get("SCRAPER_SUBJECT_PREFIX", "H2") or "H2").strip().upper()
    return f"{prefix} {base}"

_ANSWER_TITLE_RE = re.compile(
    r"\b("
    r"mark(?:ing)?\s*scheme|"
    r"ms|"
    r"answer\s*key|"
    r"answer\s*sheet|"
    r"suggested\s*answers?|"
    r"examiner(?:'s|s)?\s*report"
    r")\b",
    flags=re.IGNORECASE,
)
_TITLE_STOPWORDS = {
    "the",
    "for",
    "and",
    "with",
    "from",
    "h1",
    "h2",
    "h3",
    "a",
    "levels",
    "gce",
}

_SCHOOL_TOKEN_RE = re.compile(r"^[a-z]{2,6}jc$")
_SCHOOL_ALIASES = {"ri", "mi", "hci", "rvhs", "dhs", "ejc", "njc", "vjc", "nyjc", "cjc", "acjc", "sajc", "asrjc", "yijc", "tjc"}
_SERIES_TOKEN_RE = re.compile(r"\b(prelims?|preliminary|promotion|promo|examination|exam)\b", flags=re.IGNORECASE)

def _is_answer_key_title(title: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", (title or "").lower())
    return bool(
        _ANSWER_TITLE_RE.search(title or "")
        or any(
            token in compact
            for token in (
                "markscheme",
                "answerkey",
                "answersheet",
                "suggestedanswers",
                "examinersreport",
            )
        )
    )

def _normalize_title_for_match(title: str, strip_answer_terms: bool) -> str:
    cleaned = re.sub(r"\.pdf$", "", str(title or "").strip(), flags=re.IGNORECASE)
    cleaned = cleaned.replace("&", " and ")
    if strip_answer_terms:
        cleaned = _ANSWER_TITLE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.lower())
    return " ".join(token for token in cleaned.split() if token)

def _title_tokens(title: str, strip_answer_terms: bool) -> set[str]:
    normalized = _normalize_title_for_match(title, strip_answer_terms=strip_answer_terms)
    tokens: set[str] = set()
    for token in normalized.split():
        if token.isdigit():
            tokens.add(token)
            continue
        if len(token) <= 1 or token in _TITLE_STOPWORDS:
            continue
        tokens.add(token)
    return tokens

def _extract_school_tokens(title: str) -> set[str]:
    tokens = _title_tokens(title, strip_answer_terms=False)
    schools: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if _SCHOOL_TOKEN_RE.match(lowered):
            schools.add(lowered)
        elif lowered in _SCHOOL_ALIASES:
            schools.add(lowered)
    return schools

def _extract_series_tokens(title: str) -> set[str]:
    return {token.lower() for token in _SERIES_TOKEN_RE.findall(str(title or ""))}

def _strict_match_score(question_title: str, answer_title: str) -> float:
    question_years = _extract_years(question_title)
    answer_years = _extract_years(answer_title)
    if question_years:
        if not answer_years or question_years.isdisjoint(answer_years):
            return -1.0
    elif answer_years:
        return -1.0

    question_schools = _extract_school_tokens(question_title)
    answer_schools = _extract_school_tokens(answer_title)
    if question_schools:
        if not answer_schools or question_schools.isdisjoint(answer_schools):
            return -1.0
    elif answer_schools:
        return -1.0

    question_markers = _extract_paper_markers(question_title)
    answer_markers = _extract_paper_markers(answer_title)
    if question_markers:
        if not answer_markers or question_markers.isdisjoint(answer_markers):
            return -1.0

    question_series = _extract_series_tokens(question_title)
    answer_series = _extract_series_tokens(answer_title)
    if question_series and answer_series and question_series.isdisjoint(answer_series):
        return -1.0

    question_tokens = _title_tokens(question_title, strip_answer_terms=False)
    answer_tokens = _title_tokens(answer_title, strip_answer_terms=True)
    if not question_tokens or not answer_tokens:
        return -1.0

    overlap = question_tokens & answer_tokens
    if len(overlap) < 2:
        return -1.0

    overlap_score = len(overlap) / max(len(question_tokens), len(answer_tokens))
    sequence_score = SequenceMatcher(
        None,
        _normalize_title_for_match(question_title, strip_answer_terms=False),
        _normalize_title_for_match(answer_title, strip_answer_terms=True),
    ).ratio()
    score = (0.6 * overlap_score) + (0.4 * sequence_score)

    if question_markers and answer_markers:
        score += 0.05
    if question_schools and answer_schools:
        score += 0.05
    return score

def _extract_years(title: str) -> set[str]:
    return set(re.findall(r"\b(20\d{2})\b", str(title or "")))

def _extract_paper_markers(title: str) -> set[str]:
    lowered = str(title or "").lower()
    markers = {f"p{value}" for value in re.findall(r"\b(?:paper|p)\s*([12])\b", lowered)}
    markers.update(f"csq{value}" for value in re.findall(r"\bcsq\s*([12])\b", lowered))
    return markers

def _title_match_score(question_title: str, answer_title: str) -> float:
    return _strict_match_score(question_title, answer_title)

def _build_answer_link_map(documents: list[dict]) -> dict[str, Optional[str]]:
    docs: list[dict[str, str]] = []
    for doc in documents or []:
        name = str((doc or {}).get("document_name") or "").strip()
        source_link = str((doc or {}).get("source_link") or "").strip()
        if not name:
            continue
        docs.append({"name": name, "source_link": source_link})

    question_docs = [doc for doc in docs if not _is_answer_key_title(doc["name"])]
    answer_docs = [doc for doc in docs if _is_answer_key_title(doc["name"]) and doc["source_link"]]
    if not question_docs or not answer_docs:
        return {}

    answer_link_by_question: dict[str, Optional[str]] = {}
    for question_doc in question_docs:
        scored: list[tuple[float, Optional[str]]] = []
        for answer_doc in answer_docs:
            score = _title_match_score(question_doc["name"], answer_doc["name"])
            scored.append((score, answer_doc["source_link"] or None))
        scored = sorted(scored, key=lambda row: row[0], reverse=True)
        if not scored:
            continue
        best_score, best_link = scored[0]
        if not best_link or best_score < 0.50:
            continue
        # If question title lacks paper marker (P1/P2/CSQ), require a clear winner.
        question_markers = _extract_paper_markers(question_doc["name"])
        if not question_markers and len(scored) > 1:
            second_score = scored[1][0]
            if second_score >= 0.50 and (best_score - second_score) < 0.08:
                continue
        answer_link_by_question[question_doc["name"]] = best_link
    return answer_link_by_question

def _candidate_subject_dirs(subject: str) -> list[Path]:
    requested = safe_subject_name(subject)
    all_dirs = [path for path in DOCUMENTS_ROOT.iterdir() if path.is_dir()] if DOCUMENTS_ROOT.exists() else []
    if not all_dirs:
        return []
    exact = [path for path in all_dirs if path.name == requested]
    if exact:
        return exact

    requested_base = requested.split("__", 1)[0]
    base_matches = [path for path in all_dirs if path.name.split("__", 1)[0] == requested_base]
    if base_matches:
        return base_matches
    return all_dirs

def _find_best_answer_key_path(subject: str, question_document_name: str) -> Optional[Path]:
    candidates: list[Path] = []
    for subject_dir in _candidate_subject_dirs(subject):
        answer_key_dir = subject_dir / "answer_key"
        question_paper_dir = subject_dir / "question_paper"
        question_papers_dir = subject_dir / "question_papers"

        if answer_key_dir.exists():
            candidates.extend(answer_key_dir.glob("*.pdf"))
        # Some answer docs can be misclassified into question_paper; include obvious answer-like titles.
        for folder in (question_paper_dir, question_papers_dir):
            if not folder.exists():
                continue
            for candidate in folder.glob("*.pdf"):
                if _is_answer_key_title(candidate.stem):
                    candidates.append(candidate)
    if not candidates:
        return None

    scored_candidates: list[tuple[float, Path]] = []
    for candidate in candidates:
        score = _title_match_score(question_document_name, candidate.stem)
        scored_candidates.append((score, candidate))
    scored_candidates.sort(key=lambda row: row[0], reverse=True)
    if not scored_candidates:
        return None
    best_score, best_path = scored_candidates[0]
    if best_score < 0.50:
        return None

    question_markers = _extract_paper_markers(question_document_name)
    if not question_markers and len(scored_candidates) > 1:
        second_score = scored_candidates[1][0]
        if second_score >= 0.50 and (best_score - second_score) < 0.08:
            return None

    return best_path


def resolve_source_link(document_name: str, config: ScraperConfig) -> str:
    scraper_subject = derive_scraper_subject(config.subject)
    scraper = HolyGrailScraper(
        config.category,
        scraper_subject,
        config.year,
        config.document_type,
        pages=config.pages,
    )
    scraper._ensure_documents_cached()
    target = document_name.strip()
    for doc in scraper.documents:
        name = (doc.get("document_name") or "").strip()
        lowered = re.sub(r"[^a-z0-9]+", "", name.lower())
        if any(token in lowered for token in ("markscheme", "answerkey", "answersheet", "suggestedanswers", "examinersreport")):
            continue
        if name == target:
            return doc.get("source_link") or ""
    return ""

def _cache_key_for_config(config: ScraperConfig) -> tuple:
    return (
        normalize_category(config.category),
        derive_scraper_subject(config.subject),
        normalize_year(config.year),
        config.document_type,
        config.pages,
    )

def _update_document_cache(documents: list[dict], config: ScraperConfig) -> None:
    cache_key = _cache_key_for_config(config)
    mapping: dict[str, dict[str, Any]] = {}
    answer_link_map = _build_answer_link_map(documents)
    for doc in documents or []:
        name = (doc.get("document_name") or "").strip()
        if not name:
            continue
        mapping[name] = {
            "source_link": doc.get("source_link"),
            "year": doc.get("year"),
            "answer_link": answer_link_map.get(name),
        }
    with _document_cache_lock:
        global _document_cache_key, _document_cache
        _document_cache_key = cache_key
        _document_cache = mapping

def _get_document_cache(config: ScraperConfig) -> dict[str, dict[str, Any]]:
    cache_key = _cache_key_for_config(config)
    with _document_cache_lock:
        if _document_cache_key == cache_key and _document_cache:
            return dict(_document_cache)
    return {}

def _is_document_cache_current(config: ScraperConfig) -> bool:
    cache_key = _cache_key_for_config(config)
    with _document_cache_lock:
        return _document_cache_key == cache_key and bool(_document_cache)

def get_or_create_document_cache(config: ScraperConfig) -> dict[str, dict[str, Any]]:
    cache = _get_document_cache(config)
    if cache:
        return cache
    scraper_subject = derive_scraper_subject(config.subject)
    scraper = HolyGrailScraper(
        config.category,
        scraper_subject,
        config.year,
        config.document_type,
        pages=config.pages,
    )
    try:
        asyncio.run(scraper.get_documents())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(scraper.get_documents())
        finally:
            loop.close()
    except Exception as exc:
        print(f"Scraper error (cache build): {type(exc).__name__}: {exc}")
        return {}
    _update_document_cache(scraper.documents, config)
    return _get_document_cache(config)

def normalize_year(year_value: Optional[Any]) -> int:
    if year_value is None:
        return 0
    try:
        return int(year_value)
    except (TypeError, ValueError):
        return 0

def safe_subject_name(subject: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in subject.strip()) or "unknown"

def normalize_category(category: Optional[str]) -> Optional[str]:
    if not category:
        return category
    normalized = category.strip()
    if normalized == 'GCE "A" Levels':
        return "GCE 'A' Levels"
    return normalized


def get_client_session_id(request: Request) -> Optional[str]:
    value = request.headers.get("X-Client-Session")
    if value is None:
        value = request.query_params.get("client_session_id")
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def load_syllabus_text(subject: str) -> str:
    preferred_path = SYLLABI_DIR / f"{safe_subject_name(subject)}.txt"
    fallback_path = SYLLABI_DIR / "econs.txt"
    if preferred_path.exists():
        return preferred_path.read_text(encoding="utf-8")
    if fallback_path.exists():
        return fallback_path.read_text(encoding="utf-8")
    return ""


def extract_text_from_pdf_path(file_path: str) -> str:
    model_prompt = ""
    doc = pymupdf.open(file_path)
    try:
        for page in doc:
            model_prompt += str(page.get_text()).replace("\n", " ")
            model_prompt = re.sub(r"(\.\s*)\n\[(\d+)\]", r"\1 [\2]", model_prompt)
    finally:
        doc.close()
    return model_prompt


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    model_prompt = ""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page in doc:
            model_prompt += str(page.get_text()).replace("\n", " ")
            model_prompt = re.sub(r"(\.\s*)\n\[(\d+)\]", r"\1 [\2]", model_prompt)
    finally:
        doc.close()
    return model_prompt


def build_prompt_payload(
    model_prompt: str,
    context: QuestionContext,
    syllabus_text: str,
) -> dict[str, Any]:
    prompts_header = "Syllabus: " + syllabus_text + "Text: \n"
    prompts = [prompts_header, model_prompt]
    context_payload = context.model_dump() if hasattr(context, "model_dump") else context.dict()
    return {"text": str(prompts), "context": context_payload}

def find_question_papers(subject: str, documents_root: Path = DOCUMENTS_ROOT) -> list[str]:
    subject_dir = safe_subject_name(subject)
    base_dir = documents_root / subject_dir
    files = glob.glob(str(base_dir / "question_paper" / "*.pdf"))
    if not files:
        files = glob.glob(str(base_dir / "question_papers" / "*.pdf"))
    if not files:
        files = glob.glob(str(documents_root / "*" / "question_paper" / "*.pdf"))
    if not files:
        files = glob.glob(str(documents_root / "*" / "question_papers" / "*.pdf"))
    return files

def ensure_question_papers(config: ScraperConfig) -> list[str]:
    subject_label = normalize_subject_label(config.subject_label or config.subject)
    files = find_question_papers(subject_label)
    if files:
        if _is_document_cache_current(config):
            return files
    scraper_subject = derive_scraper_subject(config.subject)
    scraper = HolyGrailScraper(
        config.category,
        scraper_subject,
        config.year,
        config.document_type,
        pages=config.pages,
    )
    try:
        documents = asyncio.run(scraper.get_documents())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            documents = loop.run_until_complete(scraper.get_documents())
        finally:
            loop.close()
    except Exception as exc:
        print(f"Scraper error: {type(exc).__name__}: {exc}")
        return []
    if documents:
        _update_document_cache(scraper.documents, config)
        scraper.download_documents(
            documents,
            download_root=str(DOCUMENTS_ROOT),
            subject_label=subject_label,
        )
    return find_question_papers(subject_label)

def _backfill_scraped_answer_links(subject: str, document_cache: dict[str, dict[str, Any]]) -> int:
    normalized_subject = normalize_subject_label(subject)
    updates = 0
    with Session(engine) as session:
        for document_name, cached in (document_cache or {}).items():
            answer_link = str((cached or {}).get("answer_link") or "").strip()
            if not answer_link:
                continue
            updated = (
                session.query(Question)
                .filter(
                    Question.subject == normalized_subject,
                    Question.document_name == document_name,
                    or_(Question.answer_link.is_(None), Question.answer_link != answer_link),
                )
                .update({Question.answer_link: answer_link}, synchronize_session=False)
            )
            updates += int(updated or 0)
        if updates:
            session.commit()
    return updates


def _upsert_question(model: type[Question] | type[UploadedQuestion], data: dict) -> None:
    if "subject" in data and data.get("subject"):
        data["subject"] = normalize_subject_label(data.get("subject"))
    if "category" in data:
        data["category"] = normalize_category(data.get("category"))
    with Session(engine) as session:
        filters = [
            model.subject == data.get("subject"),
            model.question_text == data.get("question_text"),
            model.document_name == data.get("document_name"),
        ]
        if model is UploadedQuestion:
            filters.append(model.session_id == data.get("session_id"))
        existing = (
            session.query(model)
            .filter(*filters)
            .first()
        )
        if existing:
            existing.question_type = data.get("question_type", existing.question_type)
            existing.category = data.get("category", existing.category)
            existing.chapter = data.get("chapter", existing.chapter)
            existing.marks = data.get("marks", existing.marks)
            existing.document_name = data.get("document_name", existing.document_name)
            existing.source_link = data.get("source_link", existing.source_link)
            existing.answer_link = data.get("answer_link", existing.answer_link)
        else:
            q = model(**data)
            session.add(q)
        session.commit()


def insert_question(data: dict) -> None:
    _upsert_question(Question, data)


def insert_uploaded_question(data: dict) -> None:
    _upsert_question(UploadedQuestion, data)

def list_subtopics(subject: Optional[str] = None):
    with Session(engine) as session:
        query = session.query(Subtopic)
        if subject:
            query = query.filter(Subtopic.subject == normalize_subject_label(subject))
        query = query.order_by(Subtopic.code.asc())
        return [
            {"id": s.id, "subject": s.subject, "code": s.code, "title": s.title}
            for s in query.all()
        ]

def create_subtopic(data: SubtopicCreate):
    payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    if payload.get("subject"):
        payload["subject"] = normalize_subject_label(payload["subject"])
    with Session(engine) as session:
        subtopic = Subtopic(**payload)
        session.add(subtopic)
        session.commit()
        session.refresh(subtopic)
        return {"id": subtopic.id, "subject": subtopic.subject, "code": subtopic.code, "title": subtopic.title}

def create_subtopics_bulk(payload: SubtopicBulkCreate):
    subject = normalize_subject_label(payload.subject)
    with Session(engine) as session:
        existing = {
            row[0]
            for row in session.query(Subtopic.code).filter(Subtopic.subject == subject).all()
        }
        created = 0
        seen = set(existing)
        for item in payload.subtopics:
            code = item.get("code")
            title = item.get("title")
            if not code or not title or code in seen:
                continue
            session.add(Subtopic(subject=subject, code=code, title=title))
            created += 1
            seen.add(code)
        session.commit()
    return {"created": created}


def list_collections(subject: Optional[str] = None):
    normalized_subject = normalize_subject_label(subject) if subject else None
    with Session(engine) as session:
        query = session.query(Collection)
        if normalized_subject:
            query = query.filter(Collection.subject == normalized_subject)
        query = query.order_by(Collection.name.asc())
        rows = query.all()
        collection_ids = [row.id for row in rows]
        counts: dict[int, int] = {}
        if collection_ids:
            count_rows = (
                session.query(CollectionDocument.collection_id, func.count(CollectionDocument.id))
                .filter(CollectionDocument.collection_id.in_(collection_ids))
                .group_by(CollectionDocument.collection_id)
                .all()
            )
            counts = {row[0]: int(row[1]) for row in count_rows}
        return [
            {
                "id": row.id,
                "name": row.name,
                "subject": row.subject,
                "documents_count": counts.get(row.id, 0),
            }
            for row in rows
        ]


def create_collection(payload: CollectionCreate):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name is required.")
    subject = normalize_subject_label(payload.subject.strip())
    if not subject:
        raise HTTPException(status_code=400, detail="Collection subject is required.")

    with Session(engine) as session:
        existing = (
            session.query(Collection)
            .filter(
                func.lower(Collection.name) == name.lower(),
                Collection.subject == subject,
            )
            .first()
        )
        if existing:
            return {
                "id": existing.id,
                "name": existing.name,
                "subject": existing.subject,
                "documents_count": session.query(CollectionDocument).filter(
                    CollectionDocument.collection_id == existing.id
                ).count(),
                "created": False,
            }

        row = Collection(name=name, subject=subject)
        session.add(row)
        session.commit()
        session.refresh(row)
        return {
            "id": row.id,
            "name": row.name,
            "subject": row.subject,
            "documents_count": 0,
            "created": True,
        }


def add_document_to_collection(payload: CollectionDocumentCreate):
    subject = normalize_subject_label(payload.subject.strip())
    document_name = payload.document_name.strip()
    if not document_name:
        raise HTTPException(status_code=400, detail="document_name is required.")

    with Session(engine) as session:
        collection = session.query(Collection).filter(Collection.id == payload.collection_id).first()
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found.")

        existing = (
            session.query(CollectionDocument)
            .filter(
                CollectionDocument.collection_id == payload.collection_id,
                CollectionDocument.subject == subject,
                CollectionDocument.source_type == payload.source_type,
                CollectionDocument.document_name == document_name,
            )
            .first()
        )
        if existing:
            return {"added": False}

        row = CollectionDocument(
            collection_id=payload.collection_id,
            subject=subject,
            source_type=payload.source_type,
            document_name=document_name,
        )
        session.add(row)
        session.commit()
        return {"added": True}

def set_current_context(context: QuestionContext) -> None:
    global _current_context
    with _context_lock:
        _current_context = context

def get_current_context() -> Optional[QuestionContext]:
    with _context_lock:
        return _current_context

def refresh_context_from_scraper() -> QuestionContext:
    with _scraper_config_lock:
        config = _scraper_config
    scraper_subject = derive_scraper_subject(config.subject)
    scraper = HolyGrailScraper(
        config.category,
        scraper_subject,
        config.year,
        config.document_type,
        pages=config.pages,
    )
    raw_context = scraper.get_scraper_context()
    if not isinstance(raw_context, dict):
        raise HTTPException(status_code=500, detail="Scraper context must be a dict.")
    if "question_type" not in raw_context:
        raw_context["question_type"] = "exam"
    if "source_type" not in raw_context:
        raw_context["source_type"] = "scraped"
    if raw_context.get("year") is None:
        raw_context["year"] = 0
    if not raw_context.get("document_name"):
        raise HTTPException(status_code=500, detail="Scraper context missing document_name.")
    try:
        context = QuestionContext(**raw_context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invalid scraper context: {exc}") from exc
    set_current_context(context)
    return context


@app.get("/test")
def test_connection():
    return {"status": "connected", "message": "FastAPI backend is running!"}

@app.get("/answer-key")
def get_answer_key_document(subject: str, document_name: str):
    best_match = _find_best_answer_key_path(subject, document_name)
    if best_match is None or not best_match.exists():
        raise HTTPException(status_code=404, detail="No matching answer key found.")
    return FileResponse(
        path=str(best_match),
        media_type="application/pdf",
        filename=best_match.name,
    )

@app.get("/data")
def get_data():
    with _scraper_config_lock:
        config = _scraper_config
    try:
        # Persist scraped files under DOCUMENTS_ROOT so they are reusable across requests/users.
        files = ensure_question_papers(config)
        if not files:
            raise HTTPException(status_code=404, detail="No question papers found for this subject.")

        document_cache = _get_document_cache(config)
        _backfill_scraped_answer_links(config.subject_label or config.subject, document_cache)
        syllabus_text = load_syllabus_text(config.subject_label or config.subject)

        def build_document_payload(file_path: str) -> dict:
            model_prompt = extract_text_from_pdf_path(file_path)
            document_name = Path(file_path).stem
            cached = document_cache.get(document_name, {})
            source_link = cached.get("source_link") or ""
            answer_link = cached.get("answer_link") or ""
            cached_year = cached.get("year")
            if cached_year is not None:
                try:
                    cached_year = int(cached_year)
                except (TypeError, ValueError):
                    cached_year = None
            context = QuestionContext(
                year=cached_year if cached_year is not None else normalize_year(config.year),
                subject=config.subject_label or config.subject,
                category=config.category,
                question_type="exam",
                source_link=source_link,
                answer_link=answer_link,
                document_name=document_name,
                source_type="scraped",
            )
            return build_prompt_payload(model_prompt, context, syllabus_text)

        documents_payload = [build_document_payload(path) for path in sorted(files)]
        first_payload = documents_payload[0]
        set_current_context(QuestionContext(**first_payload["context"]))
        return {
            "text": first_payload["text"],
            "context": first_payload["context"],
            "documents": documents_payload,
        }
    except HTTPException:
        raise
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print("Data pipeline error:", error_message)
        raise HTTPException(status_code=500, detail=f"Data extraction failed: {error_message}") from exc


@app.post("/uploads/question-documents/extract")
async def extract_uploaded_question_documents(
    subject: str = Form(...),
    category: str = Form("GCE 'A' Levels"),
    client_session_id: Optional[str] = Form(None),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    normalized_subject = normalize_subject_label(subject)
    normalized_category = normalize_category(category) or "GCE 'A' Levels"
    syllabus_text = load_syllabus_text(normalized_subject)
    documents_payload = []

    for file in files:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {filename}")

        file_bytes = await file.read()
        if not file_bytes:
            continue

        model_prompt = extract_text_from_pdf_bytes(file_bytes)
        document_name = Path(filename).stem or "uploaded_document"
        context = QuestionContext(
            year=0,
            subject=normalized_subject,
            category=normalized_category,
            question_type="exam",
            source_link="",
            document_name=document_name,
            source_type="uploaded",
            client_session_id=(client_session_id or None),
        )
        documents_payload.append(build_prompt_payload(model_prompt, context, syllabus_text))

    if not documents_payload:
        raise HTTPException(status_code=400, detail="No readable PDF files were uploaded.")

    first_payload = documents_payload[0]
    set_current_context(QuestionContext(**first_payload["context"]))
    return {
        "text": first_payload["text"],
        "context": first_payload["context"],
        "documents": documents_payload,
    }

@app.post("/data")
def receive_data(data: ScrapedData):
    print("Received data:", data.text)
    return {"status": "ok"}

@app.post("/context")
def receive_context(context: QuestionContext):
    if context.category:
        context.category = normalize_category(context.category)  # type: ignore[assignment]
    if context.subject:
        context.subject = normalize_subject_label(context.subject)  # type: ignore[assignment]
    set_current_context(context)
    return {"status": "ok"}

@app.post("/scraper/config")
def set_scraper_config(config: ScraperConfig):
    global _scraper_config
    if config.subject:
        config.subject = normalize_subject_label(config.subject)  # type: ignore[assignment]
    if config.subject_label:
        config.subject_label = normalize_subject_label(config.subject_label)  # type: ignore[assignment]
    if config.category:
        config.category = normalize_category(config.category)  # type: ignore[assignment]
    with _scraper_config_lock:
        _scraper_config = config
    return {"status": "ok", "config": config}

@app.get("/scraper/config")
def get_scraper_config():
    with _scraper_config_lock:
        config = _scraper_config
    return {"config": config}

@app.get("/context")
def read_context():
    context = get_current_context()
    if not context:
        return {"status": "empty"}
    return {"status": "ok", "context": context}

@app.get("/subtopics")
def get_subtopics(subject: Optional[str] = None):
    return {"subtopics": list_subtopics(subject)}

@app.post("/subtopics")
def add_subtopic(subtopic: SubtopicCreate):
    return {"subtopic": create_subtopic(subtopic)}

@app.post("/subtopics/bulk")
def add_subtopics_bulk(subtopics: SubtopicBulkCreate):
    return create_subtopics_bulk(subtopics)

@app.get("/subjects")
def get_subjects():
    with Session(engine) as session:
        rows = session.execute(select(Subtopic.subject).distinct()).all()
    subjects = sorted({normalize_subject_label(row[0]) for row in rows if row[0]})
    return {"subjects": subjects}


@app.get("/collections")
def get_collections(subject: Optional[str] = None):
    normalized_subject = normalize_subject_label(subject) if subject and subject.lower() not in {"all", "any"} else None
    return {"collections": list_collections(normalized_subject)}


@app.post("/collections")
def add_collection(payload: CollectionCreate):
    return {"collection": create_collection(payload)}


@app.post("/collections/documents")
def add_collection_document(payload: CollectionDocumentCreate):
    return add_document_to_collection(payload)

@app.get("/questions")
def get_questions(
    request: Request,
    subject: Optional[str] = None,
    category: Optional[str] = None,
    question_type: Optional[str] = None,
    search: Optional[str] = None,
    subtopic: Optional[str] = None,
    subtopics: Optional[str] = None,
    collections: Optional[str] = None,
    source_type: Optional[str] = None,
):
    client_session_id = get_client_session_id(request)
    grail_links_cache: dict[tuple[str, str], dict[str, dict[str, Optional[str]]]] = {}

    def normalize_filter(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value or value.lower() in {"all", "any"}:
            return None
        return value

    subject = normalize_filter(subject)
    category = normalize_category(normalize_filter(category))
    question_type = normalize_filter(question_type)
    search = normalize_filter(search)
    subtopic = normalize_filter(subtopic)
    subtopics = normalize_filter(subtopics)
    collections = normalize_filter(collections)
    source_type = normalize_filter(source_type)
    if source_type:
        source_type = source_type.lower()
        if source_type not in {"scraped", "uploaded"}:
            raise HTTPException(status_code=400, detail="source_type must be 'scraped' or 'uploaded'.")
    subtopic_codes: list[str] = []
    if subtopics:
        subtopic_codes = [code.strip() for code in subtopics.split(",") if code.strip()]
    if subtopic and subtopic not in subtopic_codes:
        subtopic_codes.append(subtopic)
    collection_ids: list[int] = []
    if collections:
        for raw_id in collections.split(","):
            raw_id = raw_id.strip()
            if not raw_id:
                continue
            try:
                parsed = int(raw_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid collection id.") from exc
            if parsed > 0 and parsed not in collection_ids:
                collection_ids.append(parsed)
    if subject:
        subject = normalize_subject_label(subject)

    def escape_like(value: str) -> str:
        # Escape SQL LIKE wildcards so user input is treated literally.
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    search_terms: list[str] = []
    if search:
        search_terms = [term for term in re.split(r"\s+", search.strip()) if term]

    def sanitize_http_link(value: Optional[str]) -> Optional[str]:
        cleaned = str(value or "").strip()
        if cleaned.startswith("https://") or cleaned.startswith("http://"):
            return cleaned
        return None

    def normalize_doc_lookup_key(value: str) -> str:
        stripped = re.sub(r"\.pdf$", "", str(value or "").strip(), flags=re.IGNORECASE)
        return re.sub(r"[^a-z0-9]+", "", stripped.lower())

    def get_grail_links_for_subject(row_subject: str, row_category: Optional[str]) -> dict[str, dict[str, Optional[str]]]:
        normalized_subject = normalize_subject_label(row_subject)
        normalized_category = normalize_category(row_category) or "GCE 'A' Levels"
        cache_key = (normalized_subject, normalized_category)
        if cache_key in grail_links_cache:
            return grail_links_cache[cache_key]

        try:
            pages = int(os.environ.get("ANSWER_LINK_SCRAPER_PAGES", "20"))
        except ValueError:
            pages = 20
        pages = max(1, min(20, pages))

        config = ScraperConfig(
            category=normalized_category,
            subject=normalized_subject,
            year=None,
            document_type="Exam Papers",
            pages=pages,
            subject_label=normalized_subject,
        )
        cache = get_or_create_document_cache(config)
        links_by_document: dict[str, dict[str, Optional[str]]] = {}
        for doc_name, meta in cache.items():
            links_by_document[doc_name] = {
                "source_link": sanitize_http_link(meta.get("source_link")),  # type: ignore[arg-type]
                "answer_link": sanitize_http_link(meta.get("answer_link")),  # type: ignore[arg-type]
            }
        grail_links_cache[cache_key] = links_by_document
        return links_by_document

    def find_document_meta(
        document_name: str,
        links_by_document: dict[str, dict[str, Optional[str]]],
    ) -> Optional[dict[str, Optional[str]]]:
        target = str(document_name or "").strip()
        if not target or not links_by_document:
            return None

        # Fast paths: exact/case-insensitive/exension-insensitive matches.
        if target in links_by_document:
            return links_by_document[target]

        target_lower = target.lower()
        target_no_ext_lower = re.sub(r"\.pdf$", "", target_lower, flags=re.IGNORECASE)
        for key, meta in links_by_document.items():
            key_lower = key.lower()
            key_no_ext_lower = re.sub(r"\.pdf$", "", key_lower, flags=re.IGNORECASE)
            if (
                key_lower == target_lower
                or key_no_ext_lower == target_no_ext_lower
                or key_lower == f"{target_no_ext_lower}.pdf"
            ):
                return meta

        # Normalized exact match (drops punctuation/spacing differences).
        target_norm_key = normalize_doc_lookup_key(target)
        normalized_matches = [
            meta
            for key, meta in links_by_document.items()
            if normalize_doc_lookup_key(key) == target_norm_key
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0]

        # Conservative fuzzy fallback: only return when clearly better than alternatives.
        target_norm = _normalize_title_for_match(target, strip_answer_terms=False)
        if not target_norm:
            return None
        scored: list[tuple[float, dict[str, Optional[str]]]] = []
        for key, meta in links_by_document.items():
            key_norm = _normalize_title_for_match(key, strip_answer_terms=False)
            if not key_norm:
                continue
            score = SequenceMatcher(None, target_norm, key_norm).ratio()
            if target_norm in key_norm or key_norm in target_norm:
                score += 0.08
            scored.append((score, meta))
        scored.sort(key=lambda row: row[0], reverse=True)
        if not scored:
            return None
        best_score, best_meta = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < 0.88:
            return None
        if second_score >= 0.88 and (best_score - second_score) < 0.03:
            return None
        return best_meta

    def query_question_rows(
        session: Session,
        model: type[Question] | type[UploadedQuestion],
        source_tag: str,
    ) -> list[dict[str, Any]]:
        query = session.query(model)
        if source_tag == "uploaded":
            if not client_session_id:
                return []
            query = query.filter(model.session_id == client_session_id)
        if subject:
            query = query.filter(model.subject == subject)
        if collection_ids:
            pairs_query = (
                session.query(CollectionDocument.subject, CollectionDocument.document_name)
                .filter(
                    CollectionDocument.collection_id.in_(collection_ids),
                    CollectionDocument.source_type == source_tag,
                )
                .distinct()
            )
            if subject:
                pairs_query = pairs_query.filter(CollectionDocument.subject == subject)
            subject_doc_pairs = [
                (row[0], row[1])
                for row in pairs_query.all()
                if row[0] and row[1]
            ]
            if not subject_doc_pairs:
                return []
            query = query.filter(
                or_(
                    *[
                        and_(model.subject == pair_subject, model.document_name == pair_doc)
                        for pair_subject, pair_doc in subject_doc_pairs
                    ]
                )
            )
        if category:
            query = query.filter(model.category == category)
        if question_type:
            query = query.filter(model.question_type == question_type)
        if search_terms:
            # AND semantics across terms; avoids unexpected broad matches.
            for term in search_terms[:8]:
                pattern = f"%{escape_like(term)}%"
                query = query.filter(model.question_text.ilike(pattern, escape="\\"))
        if subtopic_codes:
            chapter_filters = []
            for code in subtopic_codes:
                chapter_filters.append(model.chapter.like(f"{code} %"))
                chapter_filters.append(model.chapter.like(f"{code}.%"))
            query = query.filter(or_(*chapter_filters))
        query = query.order_by(model.id.desc()).limit(200)
        rows = query.all()

        collection_names_by_doc: dict[tuple[str, str], list[str]] = {}
        document_pairs = {
            (q.subject, q.document_name)
            for q in rows
            if q.subject and q.document_name
        }
        if document_pairs:
            pair_filters = [
                and_(
                    CollectionDocument.subject == pair_subject,
                    CollectionDocument.document_name == pair_document,
                )
                for pair_subject, pair_document in document_pairs
            ]
            collection_rows = (
                session.query(
                    CollectionDocument.subject,
                    CollectionDocument.document_name,
                    Collection.name,
                )
                .join(Collection, Collection.id == CollectionDocument.collection_id)
                .filter(
                    CollectionDocument.source_type == source_tag,
                    or_(*pair_filters),
                )
                .all()
            )
            for pair_subject, pair_document, collection_name in collection_rows:
                key = (pair_subject, pair_document)
                collection_names_by_doc.setdefault(key, [])
                if collection_name and collection_name not in collection_names_by_doc[key]:
                    collection_names_by_doc[key].append(collection_name)

        def resolve_source_link_for_row(row: Any) -> Optional[str]:
            persisted = sanitize_http_link(getattr(row, "source_link", None))
            if persisted:
                return persisted
            if source_tag != "scraped":
                return None
            row_subject = (row.subject or "").strip()
            row_category = (row.category or "").strip()
            row_document_name = (row.document_name or "").strip()
            if not row_subject or not row_document_name:
                return None
            links_by_document = get_grail_links_for_subject(row_subject, row_category)
            matched_meta = find_document_meta(row_document_name, links_by_document)
            resolved = (matched_meta or {}).get("source_link")
            return sanitize_http_link(resolved)

        def resolve_answer_link_for_row(row: Any) -> Optional[str]:
            persisted = sanitize_http_link(getattr(row, "answer_link", None))
            if persisted:
                return persisted
            if source_tag != "scraped":
                return None
            row_subject = (row.subject or "").strip()
            row_category = (row.category or "").strip()
            row_document_name = (row.document_name or "").strip()
            if not row_subject or not row_document_name:
                return None
            links_by_document = get_grail_links_for_subject(row_subject, row_category)
            matched_meta = find_document_meta(row_document_name, links_by_document)
            resolved = (matched_meta or {}).get("answer_link")
            return sanitize_http_link(resolved)

        return [
            {
                "id": q.id,
                "subject": q.subject,
                "category": q.category,
                "question_type": q.question_type,
                "chapter": q.chapter,
                "question_text": q.question_text,
                "marks": q.marks,
                "source_link": resolve_source_link_for_row(q),
                "answer_link": resolve_answer_link_for_row(q),
                "document_name": q.document_name,
                "source_type": source_tag,
                "collections": sorted(
                    collection_names_by_doc.get((q.subject, q.document_name), []),
                    key=str.lower,
                ),
            }
            for q in rows
        ]

    with Session(engine) as session:
        scraped_results = (
            query_question_rows(session, Question, "scraped")
            if source_type in {None, "scraped"}
            else []
        )
        uploaded_results = (
            query_question_rows(session, UploadedQuestion, "uploaded")
            if source_type in {None, "uploaded"}
            else []
        )

    combined = sorted(
        [*scraped_results, *uploaded_results],
        key=lambda row: row.get("id", 0),
        reverse=True,
    )
    return {
        "questions": combined,
        "scraped_questions": scraped_results,
        "uploaded_questions": uploaded_results,
    }

@app.get("/questions/filters")
def get_question_filters(request: Request):
    client_session_id = get_client_session_id(request)
    with Session(engine) as session:
        scraped_subjects = {row[0] for row in session.execute(select(Question.subject).distinct()).all() if row[0]}
        uploaded_query = select(UploadedQuestion.subject)
        if client_session_id:
            uploaded_query = uploaded_query.where(UploadedQuestion.session_id == client_session_id)
        uploaded_subjects = {row[0] for row in session.execute(uploaded_query.distinct()).all() if row[0]}
        scraped_categories = {
            row[0] for row in session.execute(select(Question.category).distinct()).all() if row[0]
        }
        uploaded_category_query = select(UploadedQuestion.category)
        if client_session_id:
            uploaded_category_query = uploaded_category_query.where(UploadedQuestion.session_id == client_session_id)
        uploaded_categories = {row[0] for row in session.execute(uploaded_category_query.distinct()).all() if row[0]}
        source_counts = {
            "scraped": session.query(Question).count(),
            "uploaded": (
                session.query(UploadedQuestion)
                .filter(UploadedQuestion.session_id == client_session_id)
                .count()
                if client_session_id
                else 0
            ),
        }
    return {
        "subjects": sorted(scraped_subjects | uploaded_subjects),
        "categories": sorted(scraped_categories | uploaded_categories),
        "source_counts": source_counts,
    }

@app.post("/syllabus/extract")
async def extract_syllabus(
    subject: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    data = await file.read()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = TMP_DIR / file.filename
    temp_path.write_bytes(data)

    text = extract_clean_body_text(str(temp_path))

    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in subject.strip())
    output_path = SYLLABI_DIR / f"{safe_name}.txt"
    save_syllabus_text(text, output_path)

    return {"subject": subject, "text": text, "path": str(output_path)}

@app.post("/ai-result")
def receive_ai_result(payload: AIResultRequest):
    try:
        context = payload.context or get_current_context()
        if not context:
            context = refresh_context_from_scraper()
        if not context or not context.document_name:
            raise HTTPException(status_code=400, detail="Missing QuestionContext")
        target_source = (context.source_type or "scraped").lower()
        if target_source not in {"scraped", "uploaded"}:
            raise HTTPException(status_code=400, detail="Invalid context.source_type")
        client_session_id = (context.client_session_id or "").strip() or None
        if target_source == "uploaded" and not client_session_id:
            raise HTTPException(status_code=400, detail="Missing client_session_id for uploaded source.")
        insert_fn = insert_question if target_source == "scraped" else insert_uploaded_question
        data = payload.result

        def normalize_marks(value: Any) -> Optional[int]:
            if value is None or isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value if value > 0 else None
            if isinstance(value, float):
                return int(value) if value.is_integer() and value > 0 else None
            raw = str(value).strip()
            if not raw:
                return None
            bracket_match = re.fullmatch(r"\[(\d+)\]", raw)
            if bracket_match:
                return int(bracket_match.group(1))
            return int(raw) if raw.isdigit() else None

        # parse the exam-style questions 
        print("AI payload keys:", list(data.keys()) if isinstance(data, dict) else type(data))
        print("Context:", context)
        for row in data.get("exam", []):
            print("Exam row:", row)
            if not isinstance(row, dict):
                continue
            chapter = str(row.get("chapter") or "").strip()
            question_text = str(row.get("question") or "").strip()
            if not chapter or not question_text:
                continue
            marks = normalize_marks(row.get("marks"))
            question_type = "exam" if marks is not None else "understanding"
            if question_type == "understanding":
                print("Exam row has no valid marks; saved as understanding:", question_text[:120])
            data_json = {
                "subject": context.subject,
                "category": context.category,
                "question_type": question_type,
                "source_link": context.source_link,
                "answer_link": context.answer_link,
                "document_name": context.document_name,
                "chapter": chapter,
                "question_text": question_text,
                "marks": marks,
            }
            if target_source == "uploaded":
                data_json["session_id"] = client_session_id
            insert_fn(data_json)
        for row in data.get("understanding", []):
            print("Understanding row:", row)
            if not isinstance(row, dict):
                continue
            chapter = str(row.get("chapter") or "").strip()
            question_text = str(row.get("question") or "").strip()
            if not chapter or not question_text:
                continue
            data_json = {
                "subject": context.subject,
                "category": context.category,
                "question_type": "understanding",
                "source_link": context.source_link,
                "answer_link": context.answer_link,
                "document_name": context.document_name,
                "chapter": chapter,
                "question_text": question_text,
                "marks": None,
            }
            if target_source == "uploaded":
                data_json["session_id"] = client_session_id
            insert_fn(data_json)
        return {"status": "received", "source_type": target_source}
    except HTTPException:
        raise
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print("AI result error:", error_message)
        raise HTTPException(status_code=500, detail=error_message)
