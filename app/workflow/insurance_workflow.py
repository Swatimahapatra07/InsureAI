from typing import TypedDict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, START, END

from app.agents.user_profile_agent import UserProfilingAgent
from app.agents.policy_search_agent import PolicySearchAgent
from app.agents.policy_analysis_agent import PolicyAnalysisAgent
from app.agents.recommendation_agent import RecommendationAgent


# ==========================================================
# 1. WORKFLOW STATE
# ==========================================================

class InsuranceState(TypedDict, total=False):

    user_input: str
    user_profile: dict
    candidate_policies: Any
    ranked_policies: Any
    recommendation: dict


# ==========================================================
# 2. INITIALIZE AGENTS
#
# NOTE: PolicyComparisonAgent + MLRankingAgent have been merged
# into a single PolicyAnalysisAgent (see app/agents/policy_analysis_agent.py).
# The old pipeline ran TWO separate LLM calls per candidate policy,
# each preceded by rebuilding a FAISS index from scratch. The merged
# agent runs ONE LLM call per policy, using a precomputed per-policy
# vectorstore (see build_policy_indexes.py) and a disk cache keyed by
# UIN so a policy is only ever sent to the LLM once, ever.
# ==========================================================

print("\nInitializing InsureAI agents...")

profiling_agent = UserProfilingAgent()
search_agent = PolicySearchAgent()
analysis_agent = PolicyAnalysisAgent()
recommendation_agent = RecommendationAgent()

# How many independent policy analyses to run at once. These are
# independent Groq API calls (one per policy), so running them
# concurrently turns "N calls x latency" into roughly "1 x latency"
# instead of adding up sequentially.
MAX_PARALLEL_ANALYSES = 5

print("\nAll agents initialized successfully!")


# ==========================================================
# 3. USER PROFILING NODE
# ==========================================================

def user_profiling_node(state: InsuranceState):

    print("\n" + "=" * 70)
    print("STEP 1: USER PROFILING AGENT")
    print("=" * 70)

    profile = profiling_agent.create_profile(
        state["user_input"]
    )

    print("\nUser profile created:")
    print(profile)

    return {
        "user_profile": profile
    }


# ==========================================================
# 4. POLICY SEARCH NODE
# ==========================================================

def policy_search_node(state: InsuranceState):

    print("\n" + "=" * 70)
    print("STEP 2: POLICY SEARCH AGENT")
    print("=" * 70)

    results = search_agent.search_policies(
        state["user_profile"]
    )

    print(
        "\nCandidate policies found:",
        len(results)
    )

    for _, row in results.iterrows():

        print(
            "-",
            row.get(
                "product_name",
                "Unknown"
            )
        )

    return {
        "candidate_policies": results
    }


# ==========================================================
# 5. POLICY ANALYSIS NODE (merged comparison + ranking)
#
# Runs analyze_policy() for every candidate CONCURRENTLY instead of
# in a sequential for-loop, since each policy's analysis is fully
# independent of the others.
# ==========================================================

def policy_analysis_node(state: InsuranceState):

    print("\n" + "=" * 70)
    print("STEP 3: POLICY ANALYSIS AGENT (merged compare + rank)")
    print("=" * 70)

    policies = state["candidate_policies"]
    user_profile = state["user_profile"]

    rows = [row for _, row in policies.iterrows()]

    results = []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_ANALYSES) as executor:

        futures = {
            executor.submit(analysis_agent.analyze_policy, row, user_profile): row
            for row in rows
        }

        for future in as_completed(futures):

            row = futures[future]

            try:
                result = future.result()
            except Exception as exc:
                print(f"  [error] {row.get('product_name', 'Unknown')}: {exc}")
                continue

            print(f"  Analyzed: {result['policy']} -> {result['score']}/100")

            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\nAnalysis completed!")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['policy']} -> {result['score']}/100")

    return {
        "ranked_policies": results
    }


# ==========================================================
# 6. RECOMMENDATION NODE
# ==========================================================

def recommendation_node(state: InsuranceState):

    print("\n" + "=" * 70)
    print("STEP 4: RECOMMENDATION AGENT")
    print("=" * 70)

    recommendation = (
        recommendation_agent.generate_recommendation(
            state["user_profile"],
            state["ranked_policies"]
        )
    )

    print("\nRecommendation generated!")

    return {
        "recommendation": recommendation
    }


# ==========================================================
# 7. CREATE LANGGRAPH
# ==========================================================

workflow = StateGraph(
    InsuranceState
)


# ==========================================================
# 8. ADD NODES
# ==========================================================

workflow.add_node(
    "user_profiling",
    user_profiling_node
)

workflow.add_node(
    "policy_search",
    policy_search_node
)

workflow.add_node(
    "policy_analysis",
    policy_analysis_node
)

workflow.add_node(
    "recommendation",
    recommendation_node
)


# ==========================================================
# 9. CONNECT NODES
# ==========================================================

workflow.add_edge(
    START,
    "user_profiling"
)

workflow.add_edge(
    "user_profiling",
    "policy_search"
)

workflow.add_edge(
    "policy_search",
    "policy_analysis"
)

workflow.add_edge(
    "policy_analysis",
    "recommendation"
)

workflow.add_edge(
    "recommendation",
    END
)


# ==========================================================
# 10. COMPILE WORKFLOW
# ==========================================================

app = workflow.compile()


# ==========================================================
# 11. RUN WORKFLOW
# ==========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("INSUREAI - AGENTIC HEALTH INSURANCE SYSTEM")
    print("=" * 70)

    user_input = input(
        "\nEnter your health insurance requirements:\n"
    )

    initial_state: InsuranceState = {
        "user_input": user_input
    }

    print("\nStarting Agentic AI workflow...")

    final_state = app.invoke(
        initial_state
    )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    print("\n")
    print("=" * 70)
    print("FINAL INSURANCE RECOMMENDATION")
    print("=" * 70)

    recommendation = final_state.get(
        "recommendation"
    )

    if recommendation:

        print(
            "\nRecommended Policy:",
            recommendation.get(
                "recommended_policy",
                "Not available"
            )
        )

        print(
            "Insurer:",
            recommendation.get(
                "insurer",
                "Not available"
            )
        )

        print(
            "UIN:",
            recommendation.get(
                "uin",
                "Not available"
            )
        )

        print(
            "Suitability Score:",
            recommendation.get(
                "suitability_score",
                "Not available"
            )
        )

        print(
            "\nWhy this policy?",
            recommendation.get(
                "reason",
                "Not available"
            )
        )

        print("\nKey Benefits:")

        for benefit in recommendation.get(
            "key_benefits",
            []
        ):

            print(
                "-",
                benefit
            )

        print("\nImportant Considerations:")

        for consideration in recommendation.get(
            "important_considerations",
            []
        ):

            print(
                "-",
                consideration
            )

        print(
            "\nAlternative Policy:",
            recommendation.get(
                "alternative_policy",
                "Not available"
            )
        )

    else:

        print(
            "\nNo recommendation was generated."
        )

    print("\n")
    print("=" * 70)
    print("INSUREAI WORKFLOW COMPLETED")
    print("=" * 70)
