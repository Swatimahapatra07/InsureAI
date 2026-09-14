"""
Shared utilities for locating a policy's source data (its row in the
CSV and its chunk file on disk). Used by both the offline precompute
script (build_policy_indexes.py) and PolicyAnalysisAgent, so the
lookup logic only lives in one place instead of being duplicated
across agents (as it previously was in PolicyComparisonAgent and
MLRankingAgent).
"""

import os


CHUNK_FOLDER = "data/processed/chunks"


def find_policy_row(df, policy_name):
    """
    Find a policy's row in the dataframe by product name, with a
    fallback to word-overlap matching if there's no direct substring
    match. Returns a pandas Series or None.
    """

    policy_name_lower = str(policy_name).strip().lower()

    matches = df[
        df["product_name"]
        .astype(str)
        .str.lower()
        .str.contains(policy_name_lower, na=False, regex=False)
    ]

    if len(matches) == 0:

        policy_words = [
            word for word in policy_name_lower.split() if len(word) > 3
        ]

        if policy_words:

            for index, row in df.iterrows():

                product = str(row.get("product_name", "")).lower()

                if all(word in product for word in policy_words):

                    matches = df.iloc[[index]]
                    break

    if len(matches) == 0:
        return None

    return matches.iloc[0]


def find_chunk_file(uin, product_name, chunk_folder=CHUNK_FOLDER):
    """
    Locate the chunk file on disk for a given policy. Tries UIN
    matching first (most reliable), then falls back to matching all
    significant words in the product name.
    """

    if not os.path.exists(chunk_folder):
        return None

    possible_files = os.listdir(chunk_folder)

    uin = str(uin).strip()

    if uin and uin.lower() != "nan":

        for file_name in possible_files:

            if uin.lower() in file_name.lower():
                return file_name

    product_words = [
        word for word in str(product_name).lower().split() if len(word) > 3
    ]

    for file_name in possible_files:

        file_lower = file_name.lower()

        if product_words and all(word in file_lower for word in product_words):
            return file_name

    return None


def load_policy_chunks(chunk_file, chunk_folder=CHUNK_FOLDER):
    """Load and split a policy's chunk file into a list of raw text chunks."""

    if not chunk_file:
        return []

    chunk_path = os.path.join(chunk_folder, chunk_file)

    if not os.path.exists(chunk_path):
        return []

    with open(chunk_path, "r", encoding="utf-8") as file:
        text = file.read()

    raw_chunks = text.split("===== CHUNK ")

    documents = []

    for chunk in raw_chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        parts = chunk.split("=====", 1)

        chunk_text = parts[1].strip() if len(parts) == 2 else chunk

        if chunk_text:
            documents.append(chunk_text)

    return documents


def safe_index_name(uin, chunk_file):
    """Build a filesystem-safe folder name to key a per-policy vectorstore by."""

    key = uin if uin and str(uin).lower() != "nan" else chunk_file

    key = str(key).strip()

    safe = "".join(
        char if (char.isalnum() or char in "-_.") else "_" for char in key
    )

    return safe or "unknown_policy"
