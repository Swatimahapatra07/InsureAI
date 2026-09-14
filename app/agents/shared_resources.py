"""
Shared LLM / embeddings singletons.

Previously every agent (UserProfilingAgent, PolicyComparisonAgent,
MLRankingAgent, RecommendationAgent) created its own ChatGroq client
and its own HuggingFaceEmbeddings model. That meant the sentence-
transformer model was loaded into memory 2-3 separate times at
startup, and made it awkward to share things like a single cached
per-policy vectorstore across agents.

This module gives every agent the SAME instances (lazy-initialized,
created once, reused everywhere). Agents still accept an optional
`llm=` / `embeddings=` constructor argument so standalone scripts /
tests that instantiate an agent on its own keep working unchanged.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

_llm = None
_embeddings = None


def get_shared_llm():
    """Return a single shared ChatGroq instance (created on first use)."""

    global _llm

    if _llm is None:

        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in .env")

        _llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0
        )

    return _llm


def get_shared_embeddings():
    """Return a single shared HuggingFaceEmbeddings instance (created on first use)."""

    global _embeddings

    if _embeddings is None:

        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embeddings
