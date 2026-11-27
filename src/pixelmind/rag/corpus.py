"""Corpus loading and chunking (dossier §8.2).

Each chunk keeps a stable id and a reference to its source document, section and page.
Markdown is split by headings; PDF text by numbered section titles ("6.3 Modèle U-Net").
Long sections are split into paragraph windows of at most `max_chars`.
"""

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

MAX_CHARS = 1200

_MD_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_PDF_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\.?\s+([A-ZÀ-Ý][^|]{2,70})$")
_PDF_FOOTER = re.compile(r"^PIXELMIND \| Dossier d'architecture technique.*$", re.M)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    doc_type: str
    licence: str
    path: str
    sha256: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    licence: str
    source: str  # path in the repository
    section: str
    page: int | None
    text: str

    def to_payload(self) -> dict:
        return asdict(self)


def load_manifest(path: Path, base: Path) -> list[tuple[Document, Path]]:
    """Documents listed in the corpus manifest (globs expanded, sorted, deduplicated)."""
    spec = yaml.safe_load(path.read_text())
    out, seen = [], set()
    for entry in spec["documents"]:
        matches = sorted(base.glob(entry["path"]))
        if not matches:
            raise FileNotFoundError(f"corpus entry matches no file: {entry['path']}")
        for f in matches:
            rel = f.relative_to(base).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            doc = Document(
                doc_id=re.sub(r"[^a-z0-9]+", "_", f.stem.lower()).strip("_"),
                title=entry.get("title") or _md_title(f) or f.stem,
                doc_type=entry["doc_type"],
                licence=entry["licence"],
                path=rel,
                sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            )
            out.append((doc, f))
    return out


def _md_title(f: Path) -> str | None:
    if f.suffix != ".md":
        return None
    for line in f.read_text().splitlines():
        if m := _MD_HEADING.match(line):
            return m.group(2)
    return None


def _windows(paragraphs: list[str], max_chars: int) -> list[str]:
    """Greedy paragraph packing; a paragraph longer than max_chars is cut on sentences."""
    out, cur = [], ""
    for p in paragraphs:
        pieces = [p] if len(p) <= max_chars else re.split(r"(?<=[.;:])\s+", p)
        for piece in pieces:
            if cur and len(cur) + len(piece) + 2 > max_chars:
                out.append(cur)
                cur = ""
            cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        out.append(cur)
    return out


def _sections_markdown(text: str) -> list[tuple[str, int | None, str]]:
    sections, path, buf = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(path) or "(début)", None, body))
        buf.clear()

    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code  # "# comment" lines inside code blocks are not headings
        m = None if in_code else _MD_HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2)]
        else:
            buf.append(line)
    flush()
    return sections


def _is_heading(number: str, text: str, major: int) -> bool:
    """Reject numbered table rows: numbering must be sequential, titles don't end with ','."""
    if text.rstrip().endswith(","):
        return False
    top, *sub = number.split(".")
    return int(top) == major + 1 if not sub else int(top) == major


def _sections_pdf(pages: list[str]) -> list[tuple[str, int | None, str]]:
    sections, current, buf = [], "(début)", []
    page_of_section, major = 1, 0

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((current, page_of_section, body))
        buf.clear()

    for page_no, text in enumerate(pages, start=1):
        for line in _PDF_FOOTER.sub("", text).splitlines():
            line = line.strip()
            m = _PDF_HEADING.match(line)
            if m and _is_heading(m.group(1), m.group(2), major):
                flush()
                major = int(m.group(1).split(".")[0])
                current, page_of_section = f"{m.group(1)} {m.group(2).strip()}", page_no
            elif line:
                buf.append(line)
    flush()
    return sections


def read_sections(f: Path) -> list[tuple[str, int | None, str]]:
    if f.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return _sections_pdf([p.extract_text() or "" for p in PdfReader(f).pages])
    return _sections_markdown(f.read_text())


def chunk_document(doc: Document, f: Path, max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks = []
    for section, page, body in read_sections(f):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for i, text in enumerate(_windows(paragraphs, max_chars)):
            key = f"{doc.doc_id}|{section}|{i}|{text}"
            chunks.append(Chunk(
                chunk_id=hashlib.sha256(key.encode()).hexdigest()[:16],
                doc_id=doc.doc_id, title=doc.title, doc_type=doc.doc_type,
                licence=doc.licence, source=doc.path, section=section, page=page, text=text,
            ))  # fmt: skip
    return chunks


def build_corpus(manifest: Path, base: Path, max_chars: int = MAX_CHARS):
    docs = load_manifest(manifest, base)
    chunks = [c for doc, f in docs for c in chunk_document(doc, f, max_chars)]
    return [d for d, _ in docs], chunks
