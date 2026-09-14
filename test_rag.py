from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

print("Loading environment variables...")

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found in .env")
    exit()

print("Groq API key loaded successfully!")
print()


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")
print()


# --------------------------------------------------
# 3. Load FAISS vector database
# --------------------------------------------------

print("Loading FAISS vector database...")

vectorstore = FAISS.load_local(
    "data/vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")
print()


# --------------------------------------------------
# 4. Create retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

print("Retriever created successfully!")
print()


# --------------------------------------------------
# 5. Load Groq LLM
# --------------------------------------------------

print("Loading Groq LLM...")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

print("Groq LLM loaded successfully!")
print()


# --------------------------------------------------
# 6. Ask the user a question
# --------------------------------------------------

query = input("Ask a question about the health insurance policies: ")

print()
print("Searching policy documents...")
print()


# --------------------------------------------------
# 7. Retrieve relevant information
# --------------------------------------------------

documents = retriever.invoke(query)

print("Relevant policy information retrieved!")
print()


# --------------------------------------------------
# 8. Create context
# --------------------------------------------------

context = "\n\n".join(
    [
        f"Source: {doc.metadata.get('source')}\n"
        f"Content: {doc.page_content}"
        for doc in documents
    ]
)


# --------------------------------------------------
# 9. Create prompt for the LLM
# --------------------------------------------------

prompt = f"""
You are an AI health insurance assistant.

Answer the user's question using ONLY the information
provided in the policy documents below.

If the information is not available in the documents,
say that the information was not found in the available
policy documents.

Do not invent policy details.

User Question:
{query}

Policy Documents:
{context}

Give a clear and simple answer.

Also mention the policy source(s) used.
"""


# --------------------------------------------------
# 10. Generate final answer
# --------------------------------------------------

print("Generating answer using Groq...")
print()

response = llm.invoke(prompt)


# --------------------------------------------------
# 11. Display final answer
# --------------------------------------------------

print("=" * 70)
print("FINAL RAG ANSWER")
print("=" * 70)

print()
print(response.content)

print()
print("=" * 70)
print("SOURCES")
print("=" * 70)

for index, document in enumerate(documents, start=1):
    print(f"{index}. {document.metadata.get('source')}")