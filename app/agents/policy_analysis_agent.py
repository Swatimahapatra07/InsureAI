"""
This agent:
  - Loads a PRECOMPUTED per-policy FAISS index (see
    build_policy_indexes.py) instead of re-embedding at request time.
    Falls back to building + saving one on the fly if it's missing,
    so the system still works even if the precompute step wasn't run.
  - Runs ONE merged retrieval pass (one deduplicated query list).
  - Makes ONE LLM call per policy that returns both the structured
    fields (for scoring) and a narrative comparison (for display).
  - Caches the LLM result per UIN on disk. A policy's document text
    never changes, so once it's been analyzed, later requests
    (including other users) reuse the cached result and skip the LLM
    call entirely. Score is still recomputed per-request since it
    depends on the current user's profile, not on the policy text.
"""

import json
import os
import re
import threading

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.agents.shared_resources import get_shared_llm, get_shared_embeddings
from app.agents.policy_utils import (
    find_policy_row,
    find_chunk_file,
    load_policy_chunks,
    safe_index_name,
)


DATASET_PATH = "data/processed/selected_health_policies.csv"
PER_POLICY_INDEX_ROOT = "data/vectorstore/per_policy"
FEATURE_CACHE_PATH = "data/cache/policy_features_cache.json"

# One merged, deduplicated query list covering everything the old
# comparison agent (6 queries) and ranking agent (7 queries) asked for.
RETRIEVAL_QUERIES = [
    "sum insured coverage amount",
    "family floater family members eligibility",
    "premium annual premium",
    "hospitalization benefits inpatient hospitalization",
    "pre-existing disease waiting period PED",
    "room rent limit ICU limit",
    "important benefits medical coverage features",
    "exclusions waiting periods conditions not covered",
]

_cache_lock = threading.Lock()


class PolicyAnalysisAgent:

    def __init__(self, llm=None, embeddings=None):

        self.df = pd.read_csv(DATASET_PATH)

        self.llm = llm or get_shared_llm()
        self.embeddings = embeddings or get_shared_embeddings()

        self._feature_cache = self._load_feature_cache()

    # ======================================================
    # DISK CACHE (keyed by UIN — policy text doesn't change,
    # so the LLM extraction only ever needs to run once per policy)
    # ======================================================

    def _load_feature_cache(self):

        if not os.path.exists(FEATURE_CACHE_PATH):
            return {}

        try:
            with open(FEATURE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_feature_cache_entry(self, uin, entry):

        with _cache_lock:

            # Re-read in case another thread updated it concurrently.
            self._feature_cache = self._load_feature_cache()
            self._feature_cache[uin] = entry

            os.makedirs(os.path.dirname(FEATURE_CACHE_PATH), exist_ok=True)

            with open(FEATURE_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._feature_cache, f, indent=2)

    # ======================================================
    # PER-POLICY VECTORSTORE (precomputed on disk; built + saved
    # on the fly as a fallback if the precompute script wasn't run)
    # ======================================================

    def _get_policy_vectorstore(self, uin, chunk_file, product_name):

        index_name = safe_index_name(uin, chunk_file)
        index_path = os.path.join(PER_POLICY_INDEX_ROOT, index_name)

        if os.path.exists(os.path.join(index_path, "index.faiss")):
            return FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        # Fallback: not precomputed yet. Build once and persist it so
        # every future request (this policy, any user) hits the cache.
        raw_chunks = load_policy_chunks(chunk_file)

        if not raw_chunks:
            return None

        documents = [
            Document(
                page_content=chunk,
                metadata={"source": chunk_file, "policy": product_name},
            )
            for chunk in raw_chunks
        ]

        vectorstore = FAISS.from_documents(documents, self.embeddings)

        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)

        return vectorstore

    def _retrieve_context(self, vectorstore):

        relevant_documents = []

        for query in RETRIEVAL_QUERIES:
            relevant_documents.extend(
                vectorstore.similarity_search(query, k=3)
            )

        unique_documents = []
        seen = set()

        for document in relevant_documents:

            text = document.page_content.strip()

            if text not in seen:
                seen.add(text)
                unique_documents.append(document)

        return unique_documents

    # ======================================================
    # ONE MERGED LLM CALL (replaces the two separate prompts)
    # ======================================================

    def _extract_with_llm(self, product_name, insurer, uin, documents, chunk_file):

        context_parts = [
            f"\n--- Policy Chunk {i} ---\n{doc.page_content}\n"
            for i, doc in enumerate(documents, start=1)
        ]
        context = "\n".join(context_parts)

        prompt = f"""
You are a Health Insurance Policy Analysis Agent.

You are analyzing ONE specific Indian health insurance policy.

Policy Name: {product_name}
Insurer: {insurer}
UIN: {uin}

Using ONLY the provided policy chunks (no outside knowledge, no
guessing, no combining with other policies), extract the following
and return ONLY a valid JSON object with exactly these fields:

{{
    "coverage": "",
    "family_floater": "",
    "annual_premium": "",
    "hospitalization": "",
    "ped_waiting_period": "",
    "room_rent_icu": "",
    "medical_benefits": "",
    "important_exclusions": "",
    "comparison_summary": ""
}}

Rules:
- If information is not available, write "Not found".
- Preserve important numbers, percentages, durations and limits.
- "family_floater" must be "Yes", "No", or "Not found".
- "comparison_summary" should be a concise 4-6 sentence plain-English
  summary covering coverage, hospitalization, waiting periods, and
  key exclusions, suitable for showing directly to a customer.
- Do NOT use Markdown or ```json fences. Return raw JSON only.

Policy Document Chunks:
{context}
"""

        response = self.llm.invoke(prompt)
        response_text = response.content.strip()

        response_text = response_text.replace("```json", "").replace("```", "").strip()

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            response_text = match.group(0)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None

    # ======================================================
    # SCORING (unchanged logic from MLRankingAgent.calculate_score
    # deliberately kept — it depends on the CURRENT user's profile,
    # so it always runs fresh, never cached)
    # ======================================================

    def _calculate_score(self, user_profile, features):

        score = 0

        coverage_text = str(features.get("coverage", "")).lower()
        desired_coverage = user_profile.get("desired_coverage")

        if coverage_text and coverage_text != "not found":

            coverage_numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", coverage_text)

            if coverage_numbers and desired_coverage:
                try:
                    max_coverage = max(
                        float(n.replace(",", "")) for n in coverage_numbers
                    )
                    # Numbers in policy docs are often in lakhs; normalize
                    # roughly if the figure looks like a lakh-scale value.
                    if max_coverage < 1000:
                        max_coverage *= 100000
                    if max_coverage >= float(desired_coverage):
                        score += 25
                    elif max_coverage >= float(desired_coverage) * 0.5:
                        score += 12
                except ValueError:
                    pass

        family_floater = str(features.get("family_floater", "")).lower()
        policy_preference = str(user_profile.get("policy_preference", "")).lower()

        if "floater" in policy_preference and family_floater == "yes":
            score += 20
        elif "floater" not in policy_preference:
            score += 10

        budget = user_profile.get("budget")
        premium_text = str(features.get("annual_premium", "")).lower()

        if budget and premium_text != "not found":

            premium_numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", premium_text)

            if premium_numbers:
                try:
                    premium = float(premium_numbers[0].replace(",", ""))
                    if premium <= budget:
                        score += 20
                    elif premium <= budget * 1.2:
                        score += 10
                except ValueError:
                    pass

        hospitalization = str(features.get("hospitalization", "")).lower()
        if hospitalization and hospitalization != "not found":
            score += 15

        ped_text = str(features.get("ped_waiting_period", "")).lower()
        if ped_text and ped_text != "not found":
            waiting_numbers = re.findall(r"\d+", ped_text)
            if waiting_numbers:
                try:
                    months = int(waiting_numbers[0])
                    if months <= 12:
                        score += 10
                    elif months <= 24:
                        score += 7
                    elif months <= 36:
                        score += 4
                    else:
                        score += 2
                except ValueError:
                    pass

        medical_needs = str(user_profile.get("medical_needs", "")).lower()
        medical_benefits = str(features.get("medical_benefits", "")).lower()

        if medical_needs and medical_benefits and medical_benefits != "not found":
            medical_words = [w for w in medical_needs.split() if len(w) > 3]
            if any(word in medical_benefits for word in medical_words):
                score += 10

        return score

    # ======================================================
    # PUBLIC ENTRY POINT — replaces both compare_policy() and
    # extract_policy_features()+calculate_score()
    # ======================================================

    def analyze_policy(self, policy_row, user_profile):

        product_name = str(policy_row.get("product_name", "")).strip()
        insurer = str(policy_row.get("insurer", "Unknown")).strip()
        uin = str(policy_row.get("uin", "")).strip()

        cache_key = uin if uin and uin.lower() != "nan" else product_name

        cached = self._feature_cache.get(cache_key)

        if cached is not None:
            features = cached["features"]
        else:

            row = policy_row

            chunk_file = find_chunk_file(uin, product_name)

            if not chunk_file:
                return {
                    "insurer": insurer,
                    "policy": product_name,
                    "uin": uin,
                    "score": 0,
                    "features": None,
                    "comparison": "Policy document not found in dataset.",
                }

            vectorstore = self._get_policy_vectorstore(uin, chunk_file, product_name)

            if vectorstore is None:
                return {
                    "insurer": insurer,
                    "policy": product_name,
                    "uin": uin,
                    "score": 0,
                    "features": None,
                    "comparison": "No policy chunks available for analysis.",
                }

            documents = self._retrieve_context(vectorstore)

            features = self._extract_with_llm(
                product_name, insurer, uin, documents, chunk_file
            )

            if features is None:
                features = {
                    "coverage": "Not found",
                    "family_floater": "Not found",
                    "annual_premium": "Not found",
                    "hospitalization": "Not found",
                    "ped_waiting_period": "Not found",
                    "room_rent_icu": "Not found",
                    "medical_benefits": "Not found",
                    "important_exclusions": "Not found",
                    "comparison_summary": "Not available",
                }

            self._save_feature_cache_entry(
                cache_key,
                {
                    "product_name": product_name,
                    "uin": uin,
                    "features": features,
                },
            )

        score = self._calculate_score(user_profile, features)

        return {
            "insurer": insurer,
            "policy": product_name,
            "uin": uin,
            "score": score,
            "features": features,
            "comparison": features.get("comparison_summary", "Not available"),
        }


if __name__ == "__main__":

    agent = PolicyAnalysisAgent()

    user_profile = {
        "age": 25,
        "family_size": 4,
        "city": "Bhubaneswar",
        "budget": 25000,
        "desired_coverage": 1000000,
        "policy_preference": "family floater",
        "medical_needs": "good hospitalization coverage",
    }

    row = agent.df.iloc[0]

    print(f"Analyzing: {row.get('product_name')}")

    result = agent.analyze_policy(row, user_profile)

    print(json.dumps(result, indent=2))
