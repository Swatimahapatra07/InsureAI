# 🏥 InsureAI

**AI-Powered Health Insurance Recommendation System** — built with Agentic AI, RAG (Retrieval-Augmented Generation), and LangGraph.

InsureAI takes a plain-English description of a user's insurance needs (age, family size, city, budget, coverage goal, and medical requirements), profiles the user, searches a curated dataset of Indian health insurance policies, analyzes and scores each candidate against the user's profile using an LLM + RAG pipeline, and returns a ranked, explainable recommendation through a Streamlit UI.

---

## ✨ Features

- **Agentic pipeline (LangGraph):** a 4-step state graph — *User Profiling → Policy Search → Policy Analysis → Recommendation* — orchestrates the end-to-end flow.
- **RAG-based policy analysis:** each policy's terms are chunked, embedded, and stored in a per-policy FAISS vector index so the LLM only sees the most relevant clauses instead of the entire policy document.
- **Precomputed vector indexes:** `build_policy_indexes.py` builds per-policy FAISS indexes offline; `PolicyAnalysisAgent` loads them at query time (falling back to building one on the fly if missing), so requests stay fast.
- **Disk caching:** LLM analysis results are cached per policy UIN (`data/cache/policy_features_cache.json`), so a policy is only ever sent to the LLM once — later requests (including from other users) reuse the cached result.
- **Concurrent analysis:** candidate policies are analyzed in parallel (`ThreadPoolExecutor`) rather than sequentially, so overall latency is close to that of a single LLM call instead of scaling with the number of candidates.
- **Streamlit UI:** a simple web form collects user requirements and displays the top recommendation plus ranked alternatives with expandable details (coverage, family floater, hospitalization, PED waiting period, room rent/ICU limits, exclusions).
- **Data pipeline scripts:** utilities to filter the raw IRDAI product list, download policy PDFs, extract text, chunk it, and build embeddings — so the whole dataset can be rebuilt from scratch.

---

## 🧠 How it works (architecture)

```
User input (Streamlit form)
        │
        ▼
┌─────────────────────┐
│ 1. User Profiling    │  UserProfilingAgent  → extracts structured
│    Agent             │  profile (age, family size, budget, etc.)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ 2. Policy Search     │  PolicySearchAgent → filters/searches the
│    Agent             │  policy dataset for candidate matches
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ 3. Policy Analysis   │  PolicyAnalysisAgent → RAG retrieval over
│    Agent             │  per-policy FAISS index + one LLM call per
│    (compare + rank)  │  policy → structured features + score
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ 4. Recommendation    │  RecommendationAgent → picks the best
│    Agent             │  policy and explains why, with alternatives
└─────────────────────┘
        │
        ▼
   Streamlit results page
```

The graph and state (`InsuranceState`) are defined in `app/workflow/insurance_workflow.py`.

---

## 📁 Project structure

```
InsureAI-main/
├── insureai_ui.py                  # Streamlit app (entry point)
├── app/
│   ├── agents/
│   │   ├── user_profile_agent.py       # Extracts a structured profile from free text
│   │   ├── policy_search_agent.py      # Searches/filters the policy dataset
│   │   ├── policy_analysis_agent.py    # RAG-based comparison + scoring (merged agent)
│   │   ├── recommendation_agent.py     # Generates the final recommendation
│   │   ├── policy_utils.py             # Shared lookup helpers (row/chunk file matching)
│   │   └── shared_resources.py         # Shared ChatGroq LLM + HuggingFace embeddings
│   └── workflow/
│       └── insurance_workflow.py       # LangGraph StateGraph wiring the agents together
│
├── filter_policies.py              # Filters raw IRDAI product list → curated policy set
├── download_policies.py            # Downloads policy PDF documents
├── extract_policy_text.py          # Extracts raw text from downloaded PDFs
├── chunk_policies.py                # Splits policy text into chunks for embedding
├── create_embeddings.py            # Builds the main FAISS vector store
├── build_policy_indexes.py         # Precomputes per-policy FAISS indexes
├── check_dataset.py                # Sanity checks on the processed dataset
├── check_policy_text.py            # Sanity checks on extracted policy text
├── test_groq.py                    # Quick test of the Groq LLM connection
├── test_retriever.py               # Quick test of the vector retriever
├── test_rag.py                     # End-to-end RAG pipeline smoke test
│
├── data/
│   ├── raw/policies/               # Downloaded policy PDFs
│   ├── processed/
│   │   ├── irdai_health_products.csv       # Raw IRDAI product catalog
│   │   ├── selected_health_policies.csv    # Filtered/curated policy dataset
│   │   ├── policy_text/                    # Extracted plain-text policies
│   │   └── chunks/                         # Chunked policy text for embedding
│   ├── vectorstore/
│   │   ├── index.faiss / index.pkl         # Main FAISS index
│   │   └── per_policy/                     # One FAISS index per policy (UIN)
│   └── cache/
│       └── policy_features_cache.json      # Cached LLM analysis results per UIN
│
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Requirements

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (used for the LLM via `langchain_groq`)

---

## 🚀 Setup

1. **Clone / unzip the project** and move into the project folder:
   ```bash
   cd InsureAI-main
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables.** Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Data files.** The repo already ships with a processed dataset, chunked policy text, and precomputed FAISS indexes under `data/`, so you can run the app immediately. If you want to rebuild the dataset from scratch (e.g. after updating the source IRDAI CSV), run the pipeline scripts in order — see [Rebuilding the data pipeline](#-rebuilding-the-data-pipeline-optional) below.

---

## ▶️ Running the app

**Streamlit UI (recommended):**
```bash
streamlit run insureai_ui.py
```
Then open the URL Streamlit prints (typically `http://localhost:8501`), fill in your age, city, family size, budget, desired coverage, and medical needs, and click **Get Recommendation**.

**Command line (LangGraph workflow directly):**
```bash
python -m app.workflow.insurance_workflow
```
This prompts for your requirements in the terminal and prints the recommendation.

---

## 🔧 Rebuilding the data pipeline (optional)

Run these scripts in order if you need to regenerate the dataset and indexes from a fresh IRDAI export:

```bash
python filter_policies.py          # Clean/filter the raw IRDAI product CSV
python download_policies.py        # Download each policy's PDF document
python extract_policy_text.py      # Extract plain text from each PDF
python chunk_policies.py           # Split policy text into chunks
python create_embeddings.py        # Build the main FAISS vector store
python build_policy_indexes.py     # Precompute one FAISS index per policy
```

Helpful checks along the way:
```bash
python check_dataset.py            # Validate the processed CSV
python check_policy_text.py        # Validate extracted policy text
python test_groq.py                # Verify GROQ_API_KEY / LLM connectivity
python test_retriever.py           # Verify a vector index retrieves sensibly
python test_rag.py                 # Smoke-test the full RAG flow
```

---

## 🛠️ Tech stack

| Layer            | Technology                                   |
|-------------------|-----------------------------------------------|
| UI                | Streamlit                                     |
| Orchestration     | LangGraph                                     |
| LLM               | Groq (`openai/gpt-oss-20b` via `langchain_groq`) |
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2` via `langchain_huggingface` |
| Vector store       | FAISS (`langchain_community`)                 |
| Data handling      | pandas                                        |
| PDF parsing       | pypdf                                         |

---

## 📝 Notes

- `GROQ_API_KEY` must be set in a `.env` file — every agent that calls the LLM will raise `ValueError` if it's missing.
- The LLM and embeddings model are lazily initialized once and shared across all agents (`app/agents/shared_resources.py`) to avoid loading the embedding model multiple times.
- Policy analysis results are cached on disk per UIN — deleting `data/cache/policy_features_cache.json` forces re-analysis on the next run.
