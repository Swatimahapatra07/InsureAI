import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Folder containing extracted policy text files
TEXT_FOLDER = "data/processed/policy_text"

# Folder where chunks will be saved
CHUNK_FOLDER = "data/processed/chunks"

os.makedirs(CHUNK_FOLDER, exist_ok=True)

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Get all TXT files
text_files = [
    file_name
    for file_name in os.listdir(TEXT_FOLDER)
    if file_name.lower().endswith(".txt")
]

print("Text files found:", len(text_files))
print()

total_chunks = 0

for index, file_name in enumerate(text_files, start=1):

    print("=" * 70)
    print(f"Processing file {index}/{len(text_files)}")
    print("File:", file_name)

    file_path = os.path.join(TEXT_FOLDER, file_name)

    # Read policy text
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split text into chunks
    chunks = text_splitter.split_text(text)

    print("Characters:", len(text))
    print("Chunks created:", len(chunks))

    # Create output file
    output_file = os.path.splitext(file_name)[0] + "_chunks.txt"
    output_path = os.path.join(CHUNK_FOLDER, output_file)

    # Save chunks
    with open(output_path, "w", encoding="utf-8") as file:

        for chunk_number, chunk in enumerate(chunks, start=1):

            file.write(
                f"\n\n===== CHUNK {chunk_number} =====\n\n"
            )

            file.write(chunk)

    total_chunks += len(chunks)

print()
print("=" * 70)
print("TEXT CHUNKING COMPLETED")
print("=" * 70)

print("Policy files processed:", len(text_files))
print("Total chunks created:", total_chunks)

print()
print("Chunks saved in:")
print(CHUNK_FOLDER)