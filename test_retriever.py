from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Location of our saved vector database
VECTORSTORE_FOLDER = "data/vectorstore"

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")
print()

print("Loading FAISS vector database...")

vectorstore = FAISS.load_local(
    VECTORSTORE_FOLDER,
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully.")
print()

# Test question
query = "What is the waiting period for pre-existing diseases?"

print("User Query:")
print(query)
print()

print("Searching relevant policy information...")
print()

# Retrieve top 5 relevant chunks
results = vectorstore.similarity_search(query, k=5)

print("=" * 70)
print("RETRIEVED POLICY INFORMATION")
print("=" * 70)

for index, document in enumerate(results, start=1):

    print()
    print(f"RESULT {index}")
    print("-" * 70)

    print("Source:", document.metadata.get("source"))

    print()
    print("Content:")
    print(document.page_content[:1000])

    print()
    print("-" * 70)