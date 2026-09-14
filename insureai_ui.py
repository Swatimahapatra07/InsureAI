import streamlit as st

from app.workflow.insurance_workflow import app as insurance_app


st.set_page_config(
    page_title="InsureAI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 InsureAI")
st.subheader("AI-Powered Health Insurance Recommendation System")

st.write(
    "Get personalized health insurance recommendations "
    "using Agentic AI, RAG, LangGraph and policy ranking."
)

st.divider()

st.header("Tell us about your insurance needs")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

family_size = st.number_input(
    "Family Size",
    min_value=1,
    max_value=20,
    value=4
)

city = st.text_input(
    "City",
    value="Bhubaneswar"
)

budget = st.number_input(
    "Annual Budget (₹)",
    min_value=1000,
    value=25000,
    step=1000
)

coverage = st.number_input(
    "Desired Coverage (₹)",
    min_value=100000,
    value=1000000,
    step=100000
)

preference = st.selectbox(
    "Policy Preference",
    [
        "Family Floater",
        "Individual",
        "Senior Citizen",
        "Any"
    ]
)

medical_needs = st.text_area(
    "Medical / Coverage Requirements",
    value="Good hospitalization coverage"
)


if st.button("🔍 Get Recommendation"):

    user_input = f"""
    I am {age} years old, live in {city}, need insurance
    for a family of {family_size}, have a budget of
    {budget} rupees, want {coverage} rupees coverage,
    and prefer a {preference.lower()} policy with
    {medical_needs}.
    """

    with st.spinner("InsureAI is analyzing insurance policies..."):

        initial_state = {
            "user_input": user_input
        }

        final_state = insurance_app.invoke(initial_state)

        recommendation = final_state.get("recommendation")


    if recommendation:

        st.success("Recommendation generated successfully!")

        st.divider()

        st.header("🏆 Recommended Policy")

        st.subheader(
            recommendation.get(
                "recommended_policy",
                "Not available"
            )
        )

        st.write(
            "**Insurer:**",
            recommendation.get("insurer", "Not available")
        )

        st.write(
            "**UIN:**",
            recommendation.get("uin", "Not available")
        )

        st.metric(
            "Suitability Score",
            f"{recommendation.get('suitability_score', 'N/A')}/100"
        )

        st.subheader("Why this policy?")

        st.write(
            recommendation.get(
                "reason",
                "No explanation available."
            )
        )

        st.subheader("✅ Key Benefits")

        for benefit in recommendation.get(
            "key_benefits",
            []
        ):
            st.write("•", benefit)

        st.subheader("⚠️ Important Considerations")

        for consideration in recommendation.get(
            "important_considerations",
            []
        ):
            st.write("•", consideration)

        st.divider()

        st.header("🔄 Alternative Plans")

        ranked_policies = final_state.get(
            "ranked_policies",
            []
        )

        recommended_uin = recommendation.get("uin")

        alternatives = [
            policy for policy in ranked_policies
            if policy.get("uin") != recommended_uin
        ]

        if alternatives:

            for alt in alternatives:

                with st.expander(
                    f"{alt.get('policy', 'Unknown')} — "
                    f"{alt.get('insurer', 'Unknown')} "
                    f"({alt.get('score', 'N/A')}/100)"
                ):

                    st.write(
                        "**UIN:**",
                        alt.get("uin", "Not available")
                    )

                    st.write(
                        "**Suitability Score:**",
                        f"{alt.get('score', 'N/A')}/100"
                    )

                    st.write(
                        "**Summary:**",
                        alt.get("comparison", "Not available")
                    )

                    features = alt.get("features") or {}

                    if features:

                        st.write(
                            "**Coverage:**",
                            features.get("coverage", "Not found")
                        )

                        st.write(
                            "**Family Floater:**",
                            features.get("family_floater", "Not found")
                        )

                        st.write(
                            "**Hospitalization:**",
                            features.get("hospitalization", "Not found")
                        )

                        st.write(
                            "**PED Waiting Period:**",
                            features.get("ped_waiting_period", "Not found")
                        )

                        st.write(
                            "**Room Rent / ICU:**",
                            features.get("room_rent_icu", "Not found")
                        )

                        st.write(
                            "**Important Exclusions:**",
                            features.get("important_exclusions", "Not found")
                        )

        else:

            st.write("No alternative plans were found for this search.")

    else:

        st.error(
            "Sorry, no recommendation could be generated."
        )