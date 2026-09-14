
import os
from pypdf import PdfReader


# ============================================================
# FOLDER PATHS
# ============================================================

PDF_FOLDER = "data/raw/policies"
OUTPUT_FOLDER = "data/processed/policy_text"


# ============================================================
# CREATE OUTPUT FOLDER IF IT DOES NOT EXIST
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# GET ALL PDF FILES
# ============================================================

pdf_files = [
    file_name
    for file_name in os.listdir(PDF_FOLDER)
    if file_name.lower().endswith(".pdf")
]


print("PDF files found:", len(pdf_files))
print()


# ============================================================
# EXTRACT TEXT FROM EACH PDF
# ============================================================

successful = 0
failed = 0
skipped = 0
empty = 0


for index, pdf_file in enumerate(pdf_files, start=1):

    print("=" * 70)
    print(f"Processing PDF {index}/{len(pdf_files)}")
    print("File:", pdf_file)

    pdf_path = os.path.join(PDF_FOLDER, pdf_file)

    # Create output TXT filename
    txt_file = os.path.splitext(pdf_file)[0] + ".txt"

    txt_path = os.path.join(
        OUTPUT_FOLDER,
        txt_file
    )

    # --------------------------------------------------------
    # SKIP IF TEXT FILE ALREADY EXISTS
    # --------------------------------------------------------

    if os.path.exists(txt_path):

        print("⏭️ Text file already exists. Skipping.")

        skipped += 1
        continue


    # --------------------------------------------------------
    # EXTRACT PDF TEXT
    # --------------------------------------------------------

    try:

        reader = PdfReader(pdf_path)

        total_pages = len(reader.pages)

        print("Total pages:", total_pages)

        extracted_text = ""

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                page_text = page.extract_text()

                if page_text:
                    extracted_text += (
                        f"\n\n--- PAGE {page_number} ---\n\n"
                    )

                    extracted_text += page_text

            except Exception as page_error:

                print(
                    f"⚠️ Could not extract page "
                    f"{page_number}: {page_error}"
                )


        # ----------------------------------------------------
        # CHECK IF TEXT WAS EXTRACTED
        # ----------------------------------------------------

        extracted_text = extracted_text.strip()

        if not extracted_text:

            print("⚠️ No text could be extracted from this PDF.")

            empty += 1
            continue


        # ----------------------------------------------------
        # SAVE TEXT FILE
        # ----------------------------------------------------

        with open(
            txt_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(extracted_text)


        text_length = len(extracted_text)

        print(
            f"✅ Text extracted successfully "
            f"({text_length} characters)"
        )

        successful += 1


    except Exception as error:

        print("❌ Failed to process PDF:")
        print(error)

        failed += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PDF TEXT EXTRACTION COMPLETED")
print("=" * 70)

print("Total PDFs found :", len(pdf_files))
print("Successfully extracted :", successful)
print("Skipped :", skipped)
print("Empty / scanned PDFs :", empty)
print("Failed :", failed)

print()
print("Text files saved in:")
print(OUTPUT_FOLDER)

print()
print("=" * 70)
