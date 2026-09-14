"""
Precompute a FAISS vectorstore for EACH policy, once, offline.

WHY THIS EXISTS
----------------
Previously, PolicyComparisonAgent and MLRankingAgent each rebuilt a
temporary FAISS index (re-running the sentence-transformer embedding
model over every chunk of a policy) on EVERY user request, for EVERY
candidate policy, in BOTH agents. That's the main reason requests
were slow: the same, unchanging policy text was being re-embedded
over and over again at request time.

This script does that embedding work once, up front, and saves the
result to disk at:

    data/vectorstore/per_policy/<uin>/

PolicyAnalysisAgent then just loads the saved index (fast, no
embedding computation) instead of rebuilding it per request.

RUN THIS:
    - once, after chunk_policies.py has produced
      data/processed/chunks/*.txt
    - again any time you add/replace a policy's chunk file

    python build_policy_indexes.py
"""

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.agents.shared_resources import get_shared_embeddings
from app.agents.policy_utils import (
    find_chunk_file,
    load_policy_chunks,
    safe_index_name,
)


DATASET_PATH = "data/processed/selected_health_policies.csv"
OUTPUT_ROOT = "data/vectorstore/per_policy"


def build_all_indexes():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print(f"Loaded {len(df)} policies from {DATASET_PATH}")

    embeddings = get_shared_embeddings()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    built = 0
    skipped = 0

    for _, row in df.iterrows():

        uin = str(row.get("uin", "")).strip()
        product_name = str(row.get("product_name", "")).strip()

        chunk_file = find_chunk_file(uin, product_name)

        if not chunk_file:
            print(f"  [skip] No chunk file found for: {product_name}")
            skipped += 1
            continue

        raw_chunks = load_policy_chunks(chunk_file)

        if not raw_chunks:
            print(f"  [skip] No chunks in file for: {product_name}")
            skipped += 1
            continue

        index_name = safe_index_name(uin, chunk_file)
        index_path = os.path.join(OUTPUT_ROOT, index_name)

        if os.path.exists(os.path.join(index_path, "index.faiss")):
            print(f"  [ok]   Already built: {product_name}")
            built += 1
            continue

        documents = [
            Document(
                page_content=chunk,
                metadata={"source": chunk_file, "policy": product_name},
            )
            for chunk in raw_chunks
        ]

        print(f"  [build] {product_name}  ({len(documents)} chunks)")

        vectorstore = FAISS.from_documents(documents, embeddings)

        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)

        built += 1

    print()
    print(f"Done. Built/verified: {built}. Skipped (no chunk file): {skipped}.")
    print(f"Indexes saved under: {OUTPUT_ROOT}/<uin>/")


if __name__ == "__main__":
    build_all_indexes()
