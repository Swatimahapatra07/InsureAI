
import os
import time
import pandas as pd
import requests


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/processed/selected_health_policies.csv"
OUTPUT_FOLDER = "data/raw/policies"


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# LOAD SELECTED POLICY DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Selected policies found:", len(df))
print()


# ============================================================
# DOWNLOAD EACH POLICY DOCUMENT
# ============================================================

successful = 0
failed = 0
skipped = 0

for index, row in df.iterrows():

    insurer = str(row["insurer"]).strip()
    product_name = str(row["product_name"]).strip()
    uin = str(row["uin"]).strip()
    document_url = str(row["document_url"]).strip()

    print("=" * 70)
    print(f"Policy {index + 1}/{len(df)}")
    print("Insurer:", insurer)
    print("Product:", product_name)
    print("UIN:", uin)

    # --------------------------------------------------------
    # Check document URL
    # --------------------------------------------------------

    if (
        document_url == ""
        or document_url.lower() == "nan"
        or not document_url.startswith("http")
    ):
        print("❌ No valid document URL found.")
        failed += 1
        continue

    print("URL:", document_url)

    # --------------------------------------------------------
    # Create safe filename
    # --------------------------------------------------------

    safe_filename = (
        product_name
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )

    filename = f"{uin}_{safe_filename}.pdf"

    file_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    # --------------------------------------------------------
    # Skip if already downloaded
    # --------------------------------------------------------

    if os.path.exists(file_path):

        print("⏭️ Already downloaded. Skipping.")

        skipped += 1
        continue

    # --------------------------------------------------------
    # Download PDF
    # --------------------------------------------------------

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/153.0 Safari/537.36"
            )
        }

        response = requests.get(
            document_url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        # ----------------------------------------------------
        # Check whether response contains a PDF
        # ----------------------------------------------------

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if (
            "pdf" not in content_type
            and not response.content.startswith(b"%PDF")
        ):
            print("⚠️ URL did not return a PDF.")

            failed += 1
            continue

        # ----------------------------------------------------
        # Save PDF
        # ----------------------------------------------------

        with open(file_path, "wb") as file:
            file.write(response.content)

        file_size = os.path.getsize(file_path)

        print(
            f"✅ Downloaded successfully "
            f"({file_size / 1024:.1f} KB)"
        )

        successful += 1

    except requests.exceptions.RequestException as error:

        print("❌ Download failed:")
        print(error)

        failed += 1

    except Exception as error:

        print("❌ Unexpected error:")
        print(error)

        failed += 1

    # --------------------------------------------------------
    # Small delay between downloads
    # --------------------------------------------------------

    time.sleep(1)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PDF DOWNLOAD COMPLETED")
print("=" * 70)

print("Total policies :", len(df))
print("Downloaded      :", successful)
print("Skipped         :", skipped)
print("Failed          :", failed)

print()
print("PDF folder:")
print(OUTPUT_FOLDER)

print()
print("=" * 70)