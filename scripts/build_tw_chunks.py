#!/usr/bin/env python3
"""
Build Taiwan additive RAG chunks from raw CSVs listed in data/sources_manifest.yaml.

Writes JSONL (one additive per line). With --embed, builds a local FAISS index
(long texts are split into multiple vectors sharing the same additive id).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "sources_manifest.yaml"
DEFAULT_JSONL = REPO_ROOT / "data" / "processed" / "taiwan" / "additive_chunks.jsonl"
DEFAULT_VECTOR_DIR = REPO_ROOT / "data" / "processed" / "taiwan" / "vector_store"

ADDITIVE_COL_MAP = {
    "項次": "item_no",
    "中文品名": "zh_name",
    "英文品名": "en_name",
    "使用食品範圍及限量": "scope_and_limits",
    "使用限制": "usage_restrictions",
    "類別": "category",
}

CATEGORY_COL_MAP = {
    "流水號": "category_serial",
    "類別名稱": "category_name",
    "描述": "category_description",
}


def _repo_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_tw_source(manifest_path: Path, source_id: str | None) -> dict[str, Any]:
    with manifest_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sources = data.get("sources") or []
    if not sources:
        raise SystemExit("Manifest has no `sources` entries.")
    if source_id:
        for s in sources:
            if s.get("source_id") == source_id:
                return s
        raise SystemExit(f"No source with source_id={source_id!r}.")
    for s in sources:
        if s.get("country_code") == "tw":
            return s
    raise SystemExit("No Taiwan (country_code=tw) source found; pass --source-id.")


def normalize_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    return s.strip()


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop trailing empty columns from CSVs with extra commas
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    for c in unnamed:
        if df[c].isna().all() or (df[c].astype(str).str.strip() == "").all():
            df = df.drop(columns=[c])
    for col in df.columns:
        if col in unnamed:
            continue
        df[col] = df[col].map(lambda x: normalize_cell(x) if pd.notna(x) else "")
    return df


def additive_chunk_id(country: str, zh_name: str, item_no: str) -> str:
    raw = f"{country}|{zh_name}|{item_no}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_chunk_text(
    zh_name: str,
    en_name: str,
    category: str,
    scope_and_limits: str,
    usage_restrictions: str,
    category_description: str,
) -> str:
    parts = [
        f"中文品名: {zh_name}",
        f"英文品名: {en_name}",
        f"類別: {category}",
    ]
    if scope_and_limits:
        parts.append(f"使用食品範圍及限量:\n{scope_and_limits}")
    if usage_restrictions:
        parts.append(f"使用限制:\n{usage_restrictions}")
    if category_description:
        parts.append(f"類別規則與說明:\n{category_description}")
    return "\n\n".join(parts)


def split_for_embedding(text: str, max_chars: int, overlap: int) -> list[str]:
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    step = max(1, max_chars - overlap)
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chars])
        start += step
    return chunks


def load_and_merge(
    additives_path: Path,
    categories_path: Path,
    encoding: str,
) -> pd.DataFrame:
    add_df = pd.read_csv(additives_path, encoding=encoding)
    cat_df = pd.read_csv(categories_path, encoding=encoding)
    add_df = clean_frame(add_df)
    cat_df = clean_frame(cat_df)
    add_df = add_df.rename(columns={k: v for k, v in ADDITIVE_COL_MAP.items() if k in add_df.columns})
    cat_df = cat_df.rename(columns={k: v for k, v in CATEGORY_COL_MAP.items() if k in cat_df.columns})
    missing = [c for c in ADDITIVE_COL_MAP.values() if c not in add_df.columns]
    if missing:
        raise SystemExit(f"Additives CSV missing columns after rename: {missing}")
    if "category_name" not in cat_df.columns or "category_description" not in cat_df.columns:
        raise SystemExit("Category rules CSV missing expected columns.")
    cat_df = cat_df.drop_duplicates(subset=["category_name"], keep="first")
    merged = add_df.merge(
        cat_df[["category_name", "category_description"]],
        how="left",
        left_on="category",
        right_on="category_name",
        validate="m:1",
    )
    merged["category_name"] = merged["category_name"].fillna("")
    merged["category_description"] = merged["category_description"].fillna("")
    return merged


def write_jsonl(
    merged: pd.DataFrame,
    out_path: Path,
    country: str,
    source_id: str,
    as_of_date: str,
    official_url: str,
) -> list[dict[str, Any]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as out_f:
        for _, row in merged.iterrows():
            zh_name = row["zh_name"]
            en_name = row["en_name"]
            item_no = str(row["item_no"])
            category = row["category"]
            scope = row.get("scope_and_limits", "")
            restrictions = row.get("usage_restrictions", "")
            cat_desc = row.get("category_description", "") or ""
            cid = additive_chunk_id(country, zh_name, item_no)
            text = build_chunk_text(
                zh_name, en_name, category, scope, restrictions, cat_desc
            )
            meta = {
                "country": country,
                "source_id": source_id,
                "zh_name": zh_name,
                "en_name": en_name,
                "category": category,
                "as_of_date": as_of_date,
                "official_url": official_url,
                "doc_type": "additive",
                "item_no": item_no,
            }
            rec = {"id": cid, "text": text, "metadata": meta}
            records.append(rec)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def build_faiss_index(
    records: list[dict[str, Any]],
    vector_dir: Path,
    embedding_model: str,
    max_chunk_chars: int,
    overlap: int,
) -> None:
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as e:
        raise SystemExit(
            "Embedding dependencies missing. Install requirements "
            "(langchain, langchain-community, langchain-google-genai, faiss-cpu)."
        ) from e

    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is required for --embed.")

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    docs: list[Document] = []
    for rec in records:
        base_id = rec["id"]
        meta_base = dict(rec["metadata"])
        pieces = split_for_embedding(rec["text"], max_chunk_chars, overlap)
        total = len(pieces)
        for i, piece in enumerate(pieces):
            md = {
                **meta_base,
                "chunk_id": base_id,
                "chunk_part": i,
                "chunk_total": total,
            }
            docs.append(Document(page_content=piece, metadata=md))

    vector_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 80
    store = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_no = i // batch_size + 1
        total_batches = (len(docs) + batch_size - 1) // batch_size

        print(f"Embedding batch {batch_no}/{total_batches}: {len(batch)} docs")

    if store is None:
        store = FAISS.from_documents(batch, embeddings)
    else:
        store.add_documents(batch)

    if i + batch_size < len(docs):
        print("Sleeping 65s to avoid Gemini free-tier 429...")
        time.sleep(65)

    if store is None:
        raise SystemExit("No documents to embed.")

    store.save_local(str(vector_dir))
    print(f"Wrote FAISS index to {vector_dir} ({len(docs)} vectors).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Taiwan additive chunks JSONL (+ optional FAISS).")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-id", type=str, default=None)
    parser.add_argument("--encoding", type=str, default="utf-8-sig")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--embed", action="store_true", help="Build FAISS index (needs GOOGLE_API_KEY).")
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="models/gemini-embedding-001",
        help="Google Generative AI embeddings model id.",
    )
    parser.add_argument(
        "--embed-max-chars",
        type=int,
        default=1800,
        help="Max characters per embedding segment (long additives split).",
    )
    parser.add_argument(
        "--embed-overlap",
        type=int,
        default=200,
        help="Overlap between embedding segments.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    source = load_tw_source(manifest_path, args.source_id)
    raw_paths = [_repo_path(p) for p in (source.get("raw_paths") or [])]
    if len(raw_paths) < 2:
        raise SystemExit("Taiwan source needs at least two raw_paths (additives + category rules).")

    additives_path = raw_paths[0]
    categories_path = raw_paths[1]
    for p in (additives_path, categories_path):
        if not p.is_file():
            raise SystemExit(f"Missing file: {p}")

    country = source.get("country_code") or "tw"
    source_id = source.get("source_id") or "tw_food_additive_positive_list"
    as_of_date = str(source.get("as_of_date") or "")
    official_url = str(source.get("official_url") or "")

    merged = load_and_merge(additives_path, categories_path, args.encoding)
    no_rule = merged["category"].ne("") & merged["category_description"].eq("")
    n_no_rule = int(no_rule.sum())
    if n_no_rule:
        print(
            f"Warning: {n_no_rule} additive rows have no matching category rule row.",
            file=sys.stderr,
        )

    out_jsonl = args.output_jsonl if args.output_jsonl.is_absolute() else REPO_ROOT / args.output_jsonl
    records = write_jsonl(merged, out_jsonl, country, source_id, as_of_date, official_url)
    print(f"Wrote {len(records)} chunks to {out_jsonl}")

    if args.embed:
        vdir = args.vector_dir if args.vector_dir.is_absolute() else REPO_ROOT / args.vector_dir
        build_faiss_index(
            records,
            vdir,
            args.embedding_model,
            args.embed_max_chars,
            args.embed_overlap,
        )


if __name__ == "__main__":
    main()
