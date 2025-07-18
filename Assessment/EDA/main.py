import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from autogen.agentchat import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
)
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

def gemini_call(prompt, model_name="models/gemini-1.5-flash"):
    return genai.GenerativeModel(model_name).generate_content(prompt).text

# ===== Agent Definitions =====
class DataPrepAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        df = st.session_state["df"]
        prompt = f"""You are a Data Cleaning Agent.
- Handle missing values
- Fix data types
- Remove duplicates

Dataset head:
{df.head().to_string()}

Summary stats:
{df.describe(include='all').to_string()}

Return Python code for preprocessing and a short explanation."""
        return gemini_call(prompt)

class EDAAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        df = st.session_state["df"]
        prompt = f"""You are an EDA Agent.
- Provide summary statistics
- Extract at least 3 insights
- Suggest visualizations

Dataset head:
{df.head().to_string()}"""
        return gemini_call(prompt)

class ReportGeneratorAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        insights = st.session_state.get("eda_output", "")
        prompt = f"""You are a Report Generator.
Create a clean EDA report based on insights:

{insights}

Include:
- Overview
- Key Findings
- Visual Suggestions
- Summary conclusion."""
        return gemini_call(prompt)

class CriticAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        report = st.session_state.get("report_output", "")
        prompt = f"""You are a Critic Agent.
Review the EDA report:

{report}

Comment on clarity, accuracy, completeness, and suggest improvements."""
        return gemini_call(prompt)

class ExecutorAgent(AssistantAgent):
    def generate_reply(self, messages, sender, config=None):
        code = st.session_state.get("prep_output", "")
        prompt = f"""You are an Executor Agent.
Validate the following data preprocessing code:

{code}

- Is it runnable?
- Suggest corrections if needed."""
        return gemini_call(prompt)

# ===== Admin / Proxy Agent =====
admin_agent = UserProxyAgent(
    name="Admin",
    human_input_mode="NEVER",
    code_execution_config=False  # disables Docker requirement
)

# ===== Streamlit UI =====
st.set_page_config(layout="wide", page_title="Agentic EDA", page_icon="🔍")

# Sidebar with instructions and progress
with st.sidebar:
    st.title("🧑‍💻 Agentic EDA")
    st.markdown("""
    **Instructions:**
    1. Upload a CSV file.
    2. Click **Run Agentic EDA** to start the analysis.
    3. Review each step in the tabs.
    """)
    st.markdown("---")
    st.markdown("**Progress:**")
    progress_steps = [
        "Upload CSV",
        "Data Preparation",
        "EDA Insights",
        "EDA Report",
        "Critic Feedback",
        "Executor Validation"
    ]
    for step in progress_steps:
        st.write(f"- {step}")
    st.markdown("---")
    st.info("Powered by Gemini + Autogen")

st.title("🔍 Agentic EDA with Gemini + Autogen")
st.markdown("<div style='margin-bottom: 1.5em;'>Upload a CSV file and let our multi-agent system analyze it step-by-step.</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state["df"] = df
    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        run_clicked = st.button("🚀 Run Agentic EDA", use_container_width=True)
    with col2:
        reset_clicked = st.button("🔄 Reset", use_container_width=True)
        if reset_clicked:
            for key in ["df", "prep_output", "eda_output", "report_output"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_rerun()

    if run_clicked:
        with st.spinner("Initializing agents..."):
            agents = [
                admin_agent,
                DataPrepAgent(name="DataPrep"),
                EDAAgent(name="EDA"),
                ReportGeneratorAgent(name="ReportGen"),
                CriticAgent(name="Critic"),
                ExecutorAgent(name="Executor"),
            ]
            chat = GroupChat(agents=agents, messages=[])
            manager = GroupChatManager(groupchat=chat)

        with st.spinner("Running multi-agent system..."):
            # Prepare outputs and error flags
            prep = eda_out = report = critique = exec_feedback = None
            error_flag = False
            # Data Preparation
            try:
                prep = agents[1].generate_reply([], "Admin")
                st.session_state["prep_output"] = prep
            except Exception as e:
                prep = f"Error: {e}"
                error_flag = True
            # EDA
            try:
                eda_out = agents[2].generate_reply([], "Admin") if not error_flag else "Skipped due to previous error."
                st.session_state["eda_output"] = eda_out
            except Exception as e:
                eda_out = f"Error: {e}"
                error_flag = True
            # Report
            try:
                report = agents[3].generate_reply([], "Admin") if not error_flag else "Skipped due to previous error."
                st.session_state["report_output"] = report
            except Exception as e:
                report = f"Error: {e}"
                error_flag = True
            # Critic
            try:
                critique = agents[4].generate_reply([], "Admin") if not error_flag else "Skipped due to previous error."
            except Exception as e:
                critique = f"Error: {e}"
                error_flag = True
            # Executor
            try:
                exec_feedback = agents[5].generate_reply([], "Admin") if not error_flag else "Skipped due to previous error."
            except Exception as e:
                exec_feedback = f"Error: {e}"
                error_flag = True

            # Tabs for each stage
            tabs = st.tabs([
                "🧹 Data Preparation",
                "📊 EDA Insights",
                "📄 EDA Report",
                "🧐 Critic Feedback",
                "✅ Executor Validation"
            ])
            with tabs[0]:
                st.markdown("**Python Code:**")
                st.code(prep or "No output.", language="python")
            with tabs[1]:
                st.markdown(eda_out or "No output.")
            with tabs[2]:
                st.markdown(report or "No output.")
            with tabs[3]:
                st.markdown(critique or "No output.")
            with tabs[4]:
                st.markdown(exec_feedback or "No output.")

            if error_flag:
                st.error("Some steps failed. Please check the outputs above.")
            else:
                st.success("✔️ Agentic EDA completed successfully.")
else:
    st.info("Upload a CSV file above to begin.")