
import pandas as pd

# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data\processed\irdai_health_products.csv"
OUTPUT_FILE = "data/processed/selected_health_policies.csv"


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Original policies:", len(df))


# ============================================================
# 2. REMOVE DUPLICATE UINs
# ============================================================

df = df.drop_duplicates(subset=["uin"]).copy()

print("After removing duplicate UINs:", len(df))


# ============================================================
# 3. KEEP NON-ARCHIVED PRODUCTS
# ============================================================

archive_status = (
    df["archive_status"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

df = df[
    archive_status.str.contains(
        "non-archived",
        regex=False,
        na=False
    )
].copy()

print("After archive filtering:", len(df))


# ============================================================
# 4. CREATE CLEAN TEXT COLUMNS
# ============================================================

df["product_name_clean"] = (
    df["product_name"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

df["product_type_clean"] = (
    df["type_of_product"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# 5. REMOVE UNRELATED / SPECIALIZED PRODUCTS
# ============================================================

exclude_product_words = [
    # Travel / other insurance
    "travel",
    "loan",
    "credit",
    "motor",

    # Accident insurance
    "personal accident",
    "accident",

    # Saral Suraksha / personal accident products
    "saral suraksha",

    # Riders / add-ons
    "rider",
    "add-on",
    "add on",
    "addon",

    # Hospital cash products
    "hospital cash",
    "hospicash",
    "daily cash",

    # Top-up products
    "top up",
    "top-up",
    "topup",
    "super top up",
    "super top-up",
    "supertopup",

    # Critical illness / cancer products
    "critical illness",
    "critical",
    "criti",
    "cancer protect",

    # Booster / recharge / specialized products
    "booster",
    "recharge",

    # Wellness add-ons
    "wellness addon",
]

exclude_pattern = "|".join(exclude_product_words)

df = df[
    ~df["product_name_clean"].str.contains(
        exclude_pattern,
        regex=True,
        na=False
    )
].copy()

print(
    "After removing unrelated/specialized products:",
    len(df)
)


# ============================================================
# 6. REMOVE GROUP / CORPORATE PRODUCTS
# ============================================================

group_words = [
    "group",
    "corporate",
    "employee",
    "employer",
]

group_pattern = "|".join(group_words)

# Remove group/corporate products based on product name
df = df[
    ~df["product_name_clean"].str.contains(
        group_pattern,
        regex=True,
        na=False
    )
].copy()

# Remove group products based on product type
df = df[
    ~df["product_type_clean"].str.contains(
        "group",
        regex=True,
        na=False
    )
].copy()

print("After removing group products:", len(df))


# ============================================================
# 7. REMOVE ADD-ON PRODUCTS
# ============================================================

df = df[
    ~df["product_type_clean"].str.contains(
        "add-on|add on|addon",
        regex=True,
        na=False
    )
].copy()

print("After removing add-on products:", len(df))


# ============================================================
# 8. SELECT MAIN HEALTH INSURANCE PRODUCTS
# ============================================================

preferred_keywords = [
    "health",
    "medical",
    "mediclaim",
    "family floater",
    "floater",
]

preferred_pattern = "|".join(preferred_keywords)

preferred_df = df[
    (
        df["product_name_clean"].str.contains(
            preferred_pattern,
            regex=True,
            na=False
        )
    )
    |
    (
        df["product_type_clean"].str.contains(
            preferred_pattern,
            regex=True,
            na=False
        )
    )
].copy()


# ============================================================
# 9. USE PREFERRED POLICIES
# ============================================================

if len(preferred_df) >= 30:

    df = preferred_df

else:

    print()
    print("Warning: Fewer than 30 suitable health policies found.")
    print("Using all remaining filtered policies.")

    df = df.copy()


# ============================================================
# 10. REMOVE DUPLICATES AGAIN
# ============================================================

df = df.drop_duplicates(
    subset=["uin"]
).copy()


# ============================================================
# 11. SELECT 30 POLICIES
# ============================================================

df = df.head(30).copy()

print("Selected policies:", len(df))


# ============================================================
# 12. REMOVE TEMPORARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "product_name_clean",
        "product_type_clean"
    ]
)


# ============================================================
# 13. SAVE FINAL DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved to:", OUTPUT_FILE)


# ============================================================
# 14. DISPLAY SELECTED POLICIES
# ============================================================

print()
print("Selected policies:")

print(
    df[
        [
            "insurer",
            "product_name",
            "uin",
            "type_of_product"
        ]
    ].to_string(index=False)
)


# ============================================================
# 15. COMPLETION MESSAGE
# ============================================================

print()
print("=" * 60)
print("POLICY FILTERING COMPLETED SUCCESSFULLY")
print("=" * 60)

