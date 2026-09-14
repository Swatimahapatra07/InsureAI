import os

TEXT_FOLDER = "data/processed/policy_text"

text_files = [
    file_name
    for file_name in os.listdir(TEXT_FOLDER)
    if file_name.lower().endswith(".txt")
]

print("Text files found:", len(text_files))
print()

for index, file_name in enumerate(text_files[:5], start=1):

    file_path = os.path.join(TEXT_FOLDER, file_name)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    print("=" * 70)
    print(f"File {index}: {file_name}")
    print("Characters:", len(text))
    print()
    print("First 500 characters:")
    print(text[:500])
    print("=" * 70)