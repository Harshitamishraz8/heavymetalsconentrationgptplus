import pandas as pd
import streamlit as st
import plotly.express as px
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()
print("DEBUG: API KEY LOADED =", os.getenv("GOOGLE_API_KEY"))

# ---------------------------
# LangChain imports
# ---------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# ---------------------------
# Helper function(s)
# ---------------------------
def calculate_indices(df):
    standard_limits = {"Fe (ppm)": 0.3, "As (ppb)": 10, "U (ppb)": 30}
    results = []

    for _, row in df.iterrows():
        Fe = row.get("Fe (ppm)")
        As = row.get("As (ppb)")
        U = row.get("U (ppb)")

        if pd.isna(Fe) or pd.isna(As) or pd.isna(U):
            continue

        CF_Fe = Fe / standard_limits["Fe (ppm)"]
        CF_As = As / standard_limits["As (ppb)"]
        CF_U = U / standard_limits["U (ppb)"]

        PLI = (CF_Fe * CF_As * CF_U) ** (1 / 3)
        HPI = (CF_Fe + CF_As + CF_U) / 3 * 100
        HEI = CF_Fe + CF_As + CF_U
        Cd = HEI

        status = "Safe" if HPI < 100 else "Marginal" if HPI < 200 else "Polluted"

        results.append({
            "Location": row.get("Location"),
            "Longitude": row.get("Longitude"),
            "Latitude": row.get("Latitude"),
            "Fe (ppm)": Fe,
            "As (ppb)": As,
            "U (ppb)": U,
            "HPI": HPI,
            "HEI": HEI,
            "Cd": Cd,
            "PLI": PLI,
            "Status": status
        })

    return pd.DataFrame(results)


@st.cache_data
def load_and_merge(file):
    df_raw = pd.read_csv(file)
    indices_df = calculate_indices(df_raw)

    merged = pd.merge(
        df_raw,
        indices_df[["Location", "Longitude", "Latitude", "HPI", "HEI", "Cd", "PLI", "Status"]],
        on=["Location", "Longitude", "Latitude"],
        how="left"
    )
    return merged


def setup_agent(df):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ Google API key not set. Please set GOOGLE_API_KEY in .env")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,  # opt-in explicitly
        verbose=True
    )
    return agent


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="Groundwater Pollution + Chatbot", layout="wide")
    st.title("💧 Groundwater Pollution Insights Chatbot")

    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])
    if not uploaded:
        st.info("Please upload a CSV file to proceed.")
        return

    try:
        df = load_and_merge(uploaded)
    except Exception as e:
        st.error(f"Error loading / merging data: {e}")
        return

    st.success("Data loaded & indices merged.")
    with st.expander("Show first few rows of processed data"):
        st.dataframe(df.head())

    # Setup agent
    try:
        agent = setup_agent(df)
    except Exception as e:
        st.error(f"Error setting up chatbot: {e}")
        return

    st.subheader("Ask your question")
    user_query = st.text_input("Your question about groundwater pollution (use simple English)")

    if user_query:
        with st.spinner("Thinking…"):
            try:
                resp = agent.run(user_query)
                st.markdown("**Answer:**")
                st.write(resp)
            except Exception as e:
                st.error(f"Error in agent response: {e}")

    st.markdown("### Example queries you can try:")
    st.markdown("- Which locations are polluted in 2023?")
    st.markdown("- Top 5 safe water sources by HPI in [District/State].")
    st.markdown("- Status of water at Maloya.")
    st.markdown("- Compare As contamination in two locations.")


if __name__ == "__main__":
    main()