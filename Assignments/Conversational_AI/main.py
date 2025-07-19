# competitor_analysis_agentic.py
import os
import streamlit as st
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Callable
import copy
from dotenv import load_dotenv
load_dotenv()

# --- Get Gemini API Key from environment variable ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(
    page_title="Retail Competitor Insights Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# --- Main UI Layout ---
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/892/892458.png", width=120)
    st.markdown("""
    ## Welcome to Retail Insights
    Unlock actionable intelligence on clothing store competitors in your area. Configure your analysis and get a business-ready report instantly!
    """)
    st.info("All data is generated using advanced AI agents and the Gemini LLM.")

# --- Sidebar Redesign ---
with st.sidebar:
    st.markdown("# ⚙️ Analysis Settings")
    with st.expander("Location & Scope", expanded=True):
        city_area = st.text_input("Target Area", "Koramangala, Bangalore", help="Specify the location for analysis.")
        num_competitors = st.select_slider("Competitor Count", options=list(range(3, 11)), value=5)
    with st.expander("Report Customization", expanded=True):
        detail_mode = st.selectbox("Report Depth", ["Brief", "Standard", "In-depth"], index=1)
        theme = st.radio("Theme", ["Light", "Dark", "Business"], index=2)
    st.markdown("---")
    st.success("Ready to analyze your market!")
    go_btn = st.button("🚀 Run Competitor Analysis", use_container_width=True)

# --- Helper Functions ---
def gemini_response_factory(key: str) -> Callable:
    def respond(history: List[Dict], **kwargs) -> str:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=key,
                temperature=0.3,
                max_output_tokens=2048
            )
            lc_history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in history]
            return llm.invoke(lc_history).content
        except Exception as err:
            return f"[Gemini Error] {err}"
    return respond

def build_llm_settings(key: str, with_func=True) -> Dict:
    settings = {
        "config_list": [{
            "model": "gemini-1.5-flash",
            "api_type": "google",
            "api_key": key,
        }],
        "timeout": 600
    }
    if with_func:
        settings["functions"] = [gemini_response_factory(key)]
    return settings

# --- Main App Logic ---
def run_dashboard():
    if go_btn:
        if not GEMINI_API_KEY:
            with col2:
                st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable and restart the app.")
            return
        with col2:
            st.info(f"**Analyzing competitors in:** {city_area}")
            st.info(f"**Number of competitors:** {num_competitors}")
            st.info(f"**Report style:** {detail_mode} | **Theme:** {theme}")
            with st.spinner("Gathering market intelligence..."):
                try:
                    base_cfg = build_llm_settings(GEMINI_API_KEY, with_func=True)
                    mgr_cfg = build_llm_settings(GEMINI_API_KEY, with_func=False)

                    scout_agent = AssistantAgent(
                        name="MarketScout",
                        llm_config=copy.deepcopy(base_cfg),
                        system_message=f"""
                        You are a retail market scout. Survey clothing stores in {city_area}.
                        1. List top {num_competitors} competitors
                        2. Classify their positioning (luxury, mid, budget)
                        3. Note foot traffic and peak hours
                        Use bullet points. Detail: {detail_mode}
                        """
                    )
                    advisor_agent = AssistantAgent(
                        name="StrategyAdvisor",
                        llm_config=copy.deepcopy(base_cfg),
                        system_message=f"""
                        You are a retail strategy advisor. Using MarketScout's findings:
                        1. Compare pricing and products
                        2. Spot market gaps
                        3. Suggest strategies for hours, promos, differentiation
                        Give actionable advice for {city_area}.
                        """
                    )
                    summary_agent = AssistantAgent(
                        name="ReportMaker",
                        llm_config=copy.deepcopy(base_cfg),
                        system_message=f"""
                        Prepare a business report with:
                        ## Retail Competition: {city_area}
                        ### 1. Competitor Table
                        ### 2. Market Overview
                        ### 3. Strategic Advice
                        ### 4. Summary
                        Format for business. Detail: {detail_mode}. Use markdown tables.
                        """
                    )
                    user_agent = UserProxyAgent(
                        name="Client",
                        human_input_mode="NEVER",
                        code_execution_config=False,
                        max_consecutive_auto_reply=2,
                        default_auto_reply="Continue, please."
                    )
                    chat = GroupChat(
                        agents=[user_agent, scout_agent, advisor_agent, summary_agent],
                        messages=[],
                        max_round=6,
                        speaker_selection_method="round_robin"
                    )
                    chat_mgr = GroupChatManager(
                        groupchat=chat,
                        llm_config=mgr_cfg
                    )
                    user_agent.initiate_chat(
                        chat_mgr,
                        message=f"""
                        Please create a {detail_mode.lower()} competitor report for clothing stores in {city_area}.
                        Include {num_competitors} competitors, market positioning, foot traffic, and strategy advice.
                        The report should be ready for business use.
                        """
                    )
                    st.success("Report Ready!")
                    st.divider()
                    report = None
                    for m in reversed(chat.messages):
                        if m["name"] == "ReportMaker" and "## Retail Competition" in m.get("content", ""):
                            report = m["content"]
                            break
                    if report:
                        st.markdown(report)
                        st.download_button(
                            label="Download Markdown Report",
                            data=report,
                            file_name=f"retail_competition_{city_area.replace(' ', '_')}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.warning("No final report found. Displaying conversation log:")
                        for m in chat.messages:
                            st.write(f"**{m['name']}:**")
                            st.markdown(m["content"])
                            st.divider()
                except Exception as err:
                    st.error(f"Error: {err}")
                    st.info("Check your API key and try again.")

if __name__ == "__main__":
    run_dashboard()