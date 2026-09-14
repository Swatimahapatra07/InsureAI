import os
import pickle

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Folder containing policy chunks
CHUNK_FOLDER = "data/processed/chunks"

# Folder where vector database will be saved
VECTORSTORE_FOLDER = "data/vectorstore"

os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

print("Loading policy chunks...")

documents = []

# Read all chunk files
chunk_files = [
    file_name
    for file_name in os.listdir(CHUNK_FOLDER)
    if file_name.lower().endswith(".txt")
]

print("Chunk files found:", len(chunk_files))
print()

for file_name in chunk_files:

    file_path = os.path.join(CHUNK_FOLDER, file_name)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split individual chunks
    chunks = text.split("===== CHUNK ")

    for chunk in chunks:

        if not chunk.strip():
            continue

        documents.append({
            "text": chunk,
            "source": file_name
        })

print("Total chunks loaded:", len(documents))
print()

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")
print()

print("Creating FAISS vector database...")
print("This may take some time...")

texts = [document["text"] for document in documents]

metadatas = [
    {"source": document["source"]}
    for document in documents
]

vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas
)

print()
print("FAISS vector database created successfully!")

vectorstore.save_local(VECTORSTORE_FOLDER)

print()
print("Vector database saved in:")
print(VECTORSTORE_FOLDER)

print()
print("=" * 70)
print("EMBEDDING CREATION COMPLETED")
print("=" * 70)