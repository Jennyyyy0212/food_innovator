"""Taiwan food-additive RAG helpers: manifest, staleness, FAISS store, retrieval."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "sources_manifest.yaml"
DEFAULT_VECTOR_DIR = REPO_ROOT / "data" / "processed" / "taiwan" / "vector_store"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


def _as_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    s = str(value).strip()
    if not s:
        return None
    return s[:10]


def load_sources_manifest(manifest_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if not path.is_file():
        return {"sources": []}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {"sources": []}


def get_tw_source(
    manifest: dict[str, Any] | None = None,
    *,
    manifest_path: Path | str | None = None,
    source_id: str | None = None,
) -> dict[str, Any] | None:
    m = manifest if manifest is not None else load_sources_manifest(manifest_path)
    sources = m.get("sources") or []
    if source_id:
        for s in sources:
            if isinstance(s, dict) and s.get("source_id") == source_id:
                return s
        return None
    for s in sources:
        if isinstance(s, dict) and s.get("country_code") == "tw":
            return s
    return None


def staleness_days(as_of: str | None, *, today: dt.date | None = None) -> int | None:
    if not as_of:
        return None
    try:
        base = dt.date.fromisoformat(str(as_of).strip()[:10])
    except ValueError:
        return None
    today = today or dt.date.today()
    return (today - base).days


def tw_reference_caption(source: dict[str, Any] | None) -> str:
    if not source:
        return ""
    d = _as_iso_date(source.get("as_of_date"))
    if not d:
        return ""
    return f"Reference as of {d}."


def tw_staleness_warning_message(
    source: dict[str, Any] | None,
    *,
    threshold_days: int = 365,
) -> str | None:
    if not source:
        return None
    d = _as_iso_date(source.get("as_of_date"))
    if not d:
        return None
    age = staleness_days(d)
    if age is None or age <= threshold_days:
        return None
    return (
        f"Taiwan reference data is marked as of {d} (over one year old); "
        "confirm against the official gazette / FDA source."
    )


def vector_store_dir_ready(vector_dir: Path | str | None = None) -> bool:
    vdir = Path(vector_dir) if vector_dir else DEFAULT_VECTOR_DIR
    return (vdir / "index.faiss").is_file() and (vdir / "index.pkl").is_file()


def load_tw_vector_store(
    *,
    google_api_key: str,
    vector_dir: Path | str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
):
    """Load read-only FAISS index built by ``scripts/build_tw_chunks.py --embed``."""
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    vdir = Path(vector_dir) if vector_dir else DEFAULT_VECTOR_DIR
    if not vector_store_dir_ready(vdir):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=google_api_key,
    )
    return FAISS.load_local(
        folder_path=str(vdir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def _doc_dedupe_key(meta: dict[str, Any], fallback: str) -> tuple[Any, ...]:
    chunk_id = meta.get("chunk_id")
    part = meta.get("chunk_part", 0)
    if chunk_id is not None:
        return ("id", chunk_id, part)
    return ("hash", hash(fallback))

def extract_section(text: str, start: str, end_markers: list[str]) -> str:
    if start not in text:
        return "資料不足"

    part = text.split(start, 1)[1].strip()

    for marker in end_markers:
        if marker in part:
            part = part.split(marker, 1)[0].strip()

    return part or "資料不足"

# do exact match and return the result -- pending to use
def exact_match_tw_additive(item: str, jsonl_path=None):
    import json

    jsonl_path = jsonl_path or (DEFAULT_VECTOR_DIR.parent / "additive_chunks.jsonl")
    q = str(item).strip().lower()

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            zh = str(meta.get("zh_name", "")).strip()
            en = str(meta.get("en_name", "")).strip()

            if q == zh.lower() or q == en.lower():
                text = rec.get("text", "")

                scope = extract_section(
                    text,
                    "使用食品範圍及限量:",
                    ["使用限制:", "類別規則與說明:"]
                )

                restrictions = extract_section(
                    text,
                    "使用限制:",
                    ["類別規則與說明:"]
                )

                return {
                    "國家": meta.get("country", "tw"),
                    "項目": zh,
                    "英文名稱": en,
                    "類型": meta.get("category", "資料不足"),
                    "使用狀態": "可查到法規資料",
                    "最大添加量": scope,
                    "適用食品類別": scope,
                    "使用限制": restrictions,
                    "條文或來源": f"{meta.get('official_url', '資料不足')}；item_no={meta.get('item_no', '')}",
                }

    return None


def retrieve_tw_additive_context_exact_first(
    vector_store,
    queries,
    *,
    jsonl_path=None,
    k=6,
):
    import json
    from pathlib import Path

    # load raw chunks
    jsonl_path = jsonl_path or (DEFAULT_VECTOR_DIR.parent / "additive_chunks.jsonl")

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)

    blocks = []
    matched = set()

    # 1️⃣ exact match first
    for q in queries:
        q = q.strip()
        for rec in records:
            meta = rec.get("metadata", {})
            zh = meta.get("zh_name", "")
            en = meta.get("en_name", "")

            if q == zh or q.lower() == en.lower():
                block = f"[{zh} / {en}] category={meta.get('category')} item_no={meta.get('item_no')}\n{rec.get('text')}"
                blocks.append(block)
                matched.add(q)
                break

    # 2️⃣ fallback to FAISS for remaining
    remaining = [q for q in queries if q not in matched]

    if remaining:
        faiss_text = retrieve_tw_additive_context(vector_store, remaining, k=k)
        if faiss_text:
            blocks.append(faiss_text)

    return "\n\n---\n\n".join(blocks)

def retrieve_tw_additive_context(
    vector_store,
    queries: Sequence[str],
    *,
    k: int = 6,
    max_chars: int = 120_000,
) -> str:
    """
    Run similarity search per query string, dedupe overlapping segments, format for prompts.

    Metadata is expected from ``build_tw_chunks.py`` (``chunk_id``, ``zh_name``, ``en_name``, etc.).
    """
    if not vector_store or not queries:
        return ""
    seen: set[tuple[Any, ...]] = set()
    blocks: list[str] = []
    total_len = 0

    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        try:
            docs = vector_store.similarity_search(q, k=k)
        except Exception:
            continue
        for doc in docs:
            text = (doc.page_content or "").strip()
            if not text:
                continue
            meta = dict(doc.metadata) if getattr(doc, "metadata", None) else {}
            if meta.get("country") and meta.get("country") != "tw":
                continue
            key = _doc_dedupe_key(meta, text)
            if key in seen:
                continue
            seen.add(key)
            zh = meta.get("zh_name") or ""
            en = meta.get("en_name") or ""
            cat = meta.get("category") or ""
            item_no = meta.get("item_no") or ""
            as_of = meta.get("as_of_date") or ""
            header_bits = [b for b in (zh, en) if b]
            header = " / ".join(header_bits) if header_bits else "Unknown additive"
            cite = (
                f"[{header}] category={cat} item_no={item_no} as_of={as_of} "
                f"source_id={meta.get('source_id', '')}"
            )
            block = f"{cite}\n{text}"
            if total_len + len(block) > max_chars:
                blocks.append("… (additional retrieved excerpts omitted for length)")
                return "\n\n---\n\n".join(blocks)
            blocks.append(block)
            total_len += len(block)

    return "\n\n---\n\n".join(blocks)
