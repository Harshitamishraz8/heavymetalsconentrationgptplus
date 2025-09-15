import os
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

# ---------------------------
# Load API Key (Secrets first, then .env)
# ---------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Google API key not found. Set it in Streamlit secrets or .env")

# ---------------------------
# LangChain imports
# ---------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# ---------------------------
# Helper functions
# ---------------------------
def calculate_indices(df):
    """Calculate Heavy Metal Pollution Indices (HPI, HEI, Cd, PLI)."""
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
    """Setup LangChain agent for querying the DataFrame."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        verbose=True
    )
    return agent

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="Groundwater Pollution Dashboard + Chatbot", layout="wide")
    st.title("💧 Groundwater Pollution Dashboard & Chatbot")

    uploaded_file = st.file_uploader("Upload your cleaned water quality dataset (CSV)", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to proceed.")
        return

    # Load & merge data
    try:
        df = load_and_merge(uploaded_file)
        st.success("✅ Data loaded & indices calculated")
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return

    # Show processed data
    with st.expander("📋 Show first few rows of processed data"):
        st.dataframe(df.head(), use_container_width=True)

    # -------------------
    # Charts & Plots
    # -------------------
    st.subheader("📊 Pollution Status Charts")
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(df, names="Status", title="Safe vs Unsafe Water Sources")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(df, x="Location", y="HPI", color="Status", title="HPI by Location")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Plotly Map
    st.subheader("🗺️ Geographical Visualization (Plotly)")
    fig_map = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Status",
        hover_name="Location",
        size_max=15,
        mapbox_style="carto-positron",
        zoom=4,
        title="Heavy Metal Pollution Map"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Folium Map
    st.subheader("🌐 Interactive Map (Folium)")
    m = folium.Map(
        location=[df["Latitude"].mean(), df["Longitude"].mean()],
        zoom_start=5
    )
    for _, row in df.iterrows():
        color = "green" if row["Status"] == "Safe" else "orange" if row["Status"] == "Marginal" else "red"
        popup_text = f"""
        <b>{row['Location']}</b><br>
        HPI: {row['HPI']:.2f}<br>
        HEI: {row['HEI']:.2f}<br>
        Status: {row['Status']}
        """
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)
    st_folium(m, width='100%', height=500)

    # -------------------
    # Chatbot
    # -------------------
    st.subheader("💬 Ask your question about groundwater pollution")
    user_query = st.text_input("Type your question in simple English")

    if user_query:
        try:
            agent = setup_agent(df)
            with st.spinner("Thinking..."):
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