from dotenv import load_dotenv
import json
import os
import re

from app.agents.shared_resources import get_shared_llm

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found in .env")
    exit()

print("Groq API key loaded successfully!")


# --------------------------------------------------
# 2. User Profiling Agent
# --------------------------------------------------

class UserProfilingAgent:

    def __init__(self, llm=None):

        # Uses the shared ChatGroq instance by default instead of
        # creating a brand new client (previously every agent loaded
        # its own).
        self.llm = llm or get_shared_llm()

    def create_profile(self, user_input):

        prompt = f"""
You are a User Profiling Agent for an Indian health insurance
recommendation system.

Read the user's requirements and extract the following fields:

1. age
2. family_size
3. city
4. budget
5. desired_coverage
6. policy_preference
7. medical_needs

IMPORTANT:
- Return ONLY a valid JSON object.
- Do NOT use Markdown.
- Do NOT use ```json.
- Do NOT provide explanations.
- If a value is not provided, use null.
- Keep the values exactly based on the user's input.
- Budget should be represented as a number in Indian Rupees.
- Coverage should be represented as a number in Indian Rupees.

Example format:

{{
    "age": 25,
    "family_size": 4,
    "city": "Bhubaneswar",
    "budget": 25000,
    "desired_coverage": 1000000,
    "policy_preference": "family floater",
    "medical_needs": "good hospitalization coverage"
}}

User input:
{user_input}
"""

        response = self.llm.invoke(prompt)

        response_text = response.content.strip()

        # --------------------------------------------------
        # 3. Clean possible Markdown formatting
        # --------------------------------------------------

        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

        # --------------------------------------------------
        # 4. Extract JSON object if extra text is returned
        # --------------------------------------------------

        match = re.search(r"\{.*\}", response_text, re.DOTALL)

        if match:
            response_text = match.group(0)

        # --------------------------------------------------
        # 5. Convert response to Python dictionary
        # --------------------------------------------------

        try:

            profile = json.loads(response_text)

            return profile

        except json.JSONDecodeError:

            print("\nERROR: Could not parse the LLM response as JSON.")

            print("\nRaw LLM response:")
            print(response.content)

            return None


# --------------------------------------------------
# 6. Test the User Profiling Agent
# --------------------------------------------------

if __name__ == "__main__":

    agent = UserProfilingAgent()

    print("\n" + "=" * 60)
    print("AI HEALTH INSURANCE USER PROFILING")
    print("=" * 60)

    user_input = input(
        "\nEnter your health insurance requirements:\n"
    )

    profile = agent.create_profile(user_input)

    # --------------------------------------------------
    # 7. Display profile
    # --------------------------------------------------

    if profile:

        print("\n" + "=" * 60)
        print("USER PROFILE")
        print("=" * 60)

        print(json.dumps(profile, indent=4))

        print("\nProfile created successfully!")

    else:

        print("\nProfile creation failed.")