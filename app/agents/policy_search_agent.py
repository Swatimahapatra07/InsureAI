import pandas as pd
import os


class PolicySearchAgent:

    def __init__(self):

        self.file_path = "data/processed/selected_health_policies.csv"

        # Check whether dataset exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        # Load policy dataset
        self.df = pd.read_csv(self.file_path)

        print("Policy dataset loaded successfully!")
        print("Total policies available:", len(self.df))

    def search_policies(self, user_profile):

        df = self.df.copy()

        # --------------------------------------------------
        # 1. Get text columns
        # --------------------------------------------------

        text_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        # Convert text columns safely to strings
        for column in text_columns:
            df[column] = df[column].fillna("").astype(str)

        # --------------------------------------------------
        # 2. Combine policy information
        # --------------------------------------------------

        df["search_text"] = (
            df[text_columns]
            .apply(
                lambda row: " ".join(row),
                axis=1
            )
            .str.lower()
        )

        # --------------------------------------------------
        # 3. Get user requirements
        # --------------------------------------------------

        preference = str(
            user_profile.get(
                "policy_preference",
                ""
            )
        ).lower()

        medical_needs = str(
            user_profile.get(
                "medical_needs",
                ""
            )
        ).lower()

        # --------------------------------------------------
        # 4. Search using policy preference
        # --------------------------------------------------

        if preference and preference != "none":

            preference_words = [
                word
                for word in preference.split()
                if len(word) > 2
            ]

            preference_mask = df["search_text"].apply(
                lambda text: any(
                    word in text
                    for word in preference_words
                )
            )

            preferred_policies = df[
                preference_mask
            ]

        else:

            preferred_policies = df

        # --------------------------------------------------
        # 5. If no exact preference matches,
        #    keep all policies as candidates
        # --------------------------------------------------

        if len(preferred_policies) == 0:

            preferred_policies = df

        # --------------------------------------------------
        # 6. Search medical needs
        # --------------------------------------------------

        if medical_needs:

            medical_keywords = [
                word
                for word in medical_needs.split()
                if len(word) > 3
            ]

            if medical_keywords:

                medical_mask = preferred_policies[
                    "search_text"
                ].apply(
                    lambda text: any(
                        word in text
                        for word in medical_keywords
                    )
                )

                medical_matches = preferred_policies[
                    medical_mask
                ]

                # Only filter if medical matches exist
                if len(medical_matches) > 0:

                    preferred_policies = medical_matches

        # --------------------------------------------------
        # 7. Limit results to top 10 candidates
        # --------------------------------------------------

        results = preferred_policies.head(10)

        # Remove temporary search column
        results = results.drop(
            columns=["search_text"],
            errors="ignore"
        )

        return results


# --------------------------------------------------
# Test Policy Search Agent
# --------------------------------------------------

if __name__ == "__main__":

    agent = PolicySearchAgent()

    # Example profile from User Profiling Agent
    user_profile = {

        "age": 25,

        "family_size": 4,

        "city": "Bhubaneswar",

        "budget": 25000,

        "desired_coverage": 1000000,

        "policy_preference": "family floater",

        "medical_needs": "good hospitalization coverage"
    }

    print("\n" + "=" * 70)
    print("SEARCHING POLICIES")
    print("=" * 70)

    results = agent.search_policies(
        user_profile
    )

    print(
        "\nMatching policies:",
        len(results)
    )

    print("\n" + "=" * 70)
    print("POLICY SEARCH RESULTS")
    print("=" * 70)

    if len(results) == 0:

        print("\nNo matching policies found.")

    else:

        for index, row in results.iterrows():

            print(
                "\nPolicy",
                index + 1
            )

            print(
                "Insurer:",
                row.get("insurer", "N/A")
            )

            print(
                "Product:",
                row.get("product_name", "N/A")
            )

            print(
                "UIN:",
                row.get("uin", "N/A")
            )

            print(
                "Product Type:",
                row.get("type_of_product", "N/A")
            )

            print(
                "Document:",
                row.get("document_filename", "N/A")
            )

            print("-" * 70)