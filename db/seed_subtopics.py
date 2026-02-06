import argparse
import re
from pathlib import Path

from sqlalchemy.orm import Session

from db.engine import engine
from db.models import Subtopic
from db.crud import create_schema

CODE_ONLY_RE = re.compile(r"^(\d+(?:\.\d+)+)$")
CODE_WITH_TITLE_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$")


def parse_subtopics(text: str):
    lines = [line.strip() for line in text.splitlines()]
    results = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        inline = CODE_WITH_TITLE_RE.match(line)
        if inline:
            code, title = inline.group(1), inline.group(2).strip()
            results.append((code, title))
            i += 1
            continue

        if CODE_ONLY_RE.match(line):
            code = line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                title = lines[j].strip()
                results.append((code, title))
                i = j + 1
                continue

        i += 1

    return results


def seed_subtopics(subject: str, path: Path, update_existing: bool):
    text = path.read_text(encoding="utf-8")
    items = parse_subtopics(text)

    if not items:
        print(f"No subtopics found in {path}")
        return

    create_schema()
    with Session(engine) as session:
        existing = {
            row[0]: row[1]
            for row in session.query(Subtopic.code, Subtopic.title)
            .filter(Subtopic.subject == subject)
            .all()
        }

        created = 0
        updated = 0
        for code, title in items:
            if code in existing:
                if update_existing and existing[code] != title:
                    session.query(Subtopic).filter(
                        Subtopic.subject == subject,
                        Subtopic.code == code,
                    ).update({"title": title})
                    updated += 1
                continue

            session.add(Subtopic(subject=subject, code=code, title=title))
            created += 1

        session.commit()

    print(
        f"Seeded {created} subtopics for {subject} from {path}"
        + (f", updated {updated}" if update_existing else "")
    )


def auto_subject_from_path(path: Path):
    name = path.stem.lower()
    if "econs" in name:
        return "Economics"
    if "math" in name:
        return "Mathematics"
    if "phy" in name or "physics" in name:
        return "Physics"
    return None


def main():
    parser = argparse.ArgumentParser(description="Seed subtopics from syllabus text")
    parser.add_argument("path", type=Path, help="Path to syllabus .txt")
    parser.add_argument("--subject", help="Subject name for the subtopics")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update titles for existing subtopics",
    )

    args = parser.parse_args()
    subject = args.subject or auto_subject_from_path(args.path)
    if not subject:
        raise SystemExit("Subject is required. Use --subject.")

    seed_subtopics(subject, args.path, args.update)


if __name__ == "__main__":
    main()
