from dotenv import load_dotenv
import os
import json
import re

from app.agents.shared_resources import get_shared_llm

load_dotenv()


class RecommendationAgent:

    def __init__(self, llm=None):

        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in .env")

        # Uses the shared ChatGroq instance by default instead of
        # creating a brand new client (previously every agent loaded
        # its own).
        self.llm = llm or get_shared_llm()
    def _condense_policies(self, ranked_policies, top_n=5, max_field_len=150):

        def trim(text):
            text = str(text)
            return text if len(text) <= max_field_len else text[:max_field_len] + "..."

        condensed = []

        for policy in ranked_policies[:top_n]:

            features = policy.get("features") or {}

            condensed.append({
                "insurer": policy.get("insurer"),
                "policy": policy.get("policy"),
                "uin": policy.get("uin"),
                "score": policy.get("score"),
                "coverage": trim(features.get("coverage", "Not found")),
                "family_floater": features.get("family_floater", "Not found"),
                "annual_premium": trim(features.get("annual_premium", "Not found")),
                "hospitalization": trim(features.get("hospitalization", "Not found")),
                "ped_waiting_period": trim(features.get("ped_waiting_period", "Not found")),
                "room_rent_icu": trim(features.get("room_rent_icu", "Not found")),
            })

        return condensed
    def generate_recommendation(
        self,
        user_profile,
        ranked_policies
    ):

        prompt = f"""
You are a Recommendation Agent for an Indian health insurance
recommendation system.

Your task is to recommend the most suitable health insurance policy
for the user based on the user's requirements and the ranked policy
results.

IMPORTANT RULES:

1. Recommend the policy that best matches the user's requirements.
2. Do NOT claim that a policy is universally the "best policy".
3. Explain why the recommended policy is suitable.
4. Use only the information provided in the user profile and ranked
   policy data.
5. Do not invent premium, coverage, benefits, waiting periods,
   eligibility or other policy details.
6. If information is unavailable, clearly say "Not available".
7. Mention important limitations or trade-offs when relevant.
8. Return ONLY valid JSON.
9. Do NOT use Markdown.
10. Do NOT use ```json.

Required JSON format:

{{
    "recommended_policy": "",
    "insurer": "",
    "uin": "",
    "suitability_score": "",
    "reason": "",
    "key_benefits": [],
    "important_considerations": []
}}

USER PROFILE:

{json.dumps(user_profile, indent=4)}

RANKED POLICIES (top candidates, condensed):

{json.dumps(self._condense_policies(ranked_policies), indent=4)}

Now generate the recommendation.
"""

        response = self.llm.invoke(prompt)

        response_text = response.content.strip()

        # Remove markdown formatting if the model adds it
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

        # Extract JSON object
        match = re.search(
            r"\{.*\}",
            response_text,
            re.DOTALL
        )

        if match:
            response_text = match.group(0)

        try:

            recommendation = json.loads(response_text)

            return recommendation

        except json.JSONDecodeError:

            print("\nERROR: Could not parse recommendation.")

            print("\nRaw LLM response:")
            print(response.content)

            return None


if __name__ == "__main__":

    agent = RecommendationAgent()

    user_profile = {
        "age": 25,
        "family_size": 4,
        "city": "Bhubaneswar",
        "budget": 25000,
        "desired_coverage": 1000000,
        "policy_preference": "family floater",
        "medical_needs": "good hospitalization coverage"
    }

    ranked_policies = [
        {
            "rank": 1,
            "insurer": "The Oriental Insurance Company Limited",
            "policy": "Happy Family Floater Policy 2021",
            "uin": "OICHLIP22010V042223",
            "suitability_score": 72,
            "family_floater": "Yes",
            "coverage": "Rs. 1 lakh to Rs. 50 lakh",
            "hospitalization": "Hospitalization expenses covered up to Sum Insured",
            "ped_waiting_period": "48 months"
        },
        {
            "rank": 2,
            "insurer": "Star Health & Allied Insurance Co. Ltd.",
            "policy": "Family Health Optima Insurance Plan",
            "uin": "SHAHLIP22030V062122",
            "suitability_score": 27,
            "family_floater": "Not found",
            "coverage": "Rs. 1 lakh to Rs. 25 lakh",
            "hospitalization": "In-patient hospitalization covered",
            "ped_waiting_period": "24 months"
        }
    ]

    print("\n" + "=" * 70)
    print("RECOMMENDATION AGENT")
    print("=" * 70)

    recommendation = agent.generate_recommendation(
        user_profile,
        ranked_policies
    )

    if recommendation:

        print("\nFINAL RECOMMENDATION")
        print("=" * 70)

        print(
            json.dumps(
                recommendation,
                indent=4
            )
        )

        print("\nRecommendation generated successfully!")