import streamlit as st
import json
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from typing import Dict, Any
import asyncio

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')  
if not api_key:
    st.error("""
    ❌ **Google API Key Missing!**
    
    Please follow these steps to set up your API key:
    
    1. **Get a Google API Key:**
       - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
       - Create a new API key
    
    2. **Create a .env file:**
       - Create a file named `.env` in the same directory as this script
       - Add this line to the file: `GOOGLE_API_KEY=your_actual_api_key_here`
       - Replace `your_actual_api_key_here` with your real API key
    
    3. **Restart the application**
    
    **Example .env file content:**
    ```
    GOOGLE_API_KEY=AIzaSyC...your_actual_key_here
    ```
    """)
    st.stop()

config_list_gemini = [{
    "model": "gemini-1.5-flash",
    "api_key": api_key,
    "api_type": "google"
}]

# Custom CSS for modern UI
st.set_page_config(
    page_title="Financial Portfolio Manager",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .form-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .report-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    
    .status-processing {
        background: #ffc107;
        color: #000;
    }
    
    .status-success {
        background: #28a745;
        color: white;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💼 AI Financial Portfolio Manager</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Intelligent Investment Analysis with Multi-Agent Collaboration</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for quick info
with st.sidebar:
    st.markdown("### 📊 Portfolio Overview")
    st.markdown("""
    This AI-powered tool analyzes your financial profile and provides personalized investment recommendations using advanced multi-agent collaboration.
    
    **Features:**
    - 🤖 Multi-Agent Analysis
    - 📈 Personalized Strategy
    - 💡 Smart Recommendations
    - 🔄 StateFlow Management
    """)
    
    st.markdown("### 🎯 How it Works")
    st.markdown("""
    1. **Input Your Data** - Financial profile & current portfolio
    2. **AI Analysis** - Multi-agent collaboration analyzes your situation
    3. **Strategy Selection** - Growth vs Value investment approach
    4. **Recommendations** - Personalized investment suggestions
    5. **Comprehensive Report** - Detailed financial roadmap
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h3>👤 Personal Financial Profile</h3></div>', unsafe_allow_html=True)
    
    with st.form("financial_form"):
        # Personal Information
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            salary = st.text_input("💰 Annual Salary (₹)", placeholder="1200000", help="Enter your annual salary in rupees")
            age = st.number_input("🎂 Your Age", min_value=18, max_value=100, step=1, help="Your current age")
        with col1_2:
            expenses = st.text_input("💸 Annual Expenses (₹)", placeholder="500000", help="Your annual expenses in rupees")
            risk = st.selectbox("⚖️ Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], help="Your investment risk preference")
        
        goals = st.text_area("🎯 Financial Goals", placeholder="Retirement in 20 years, buying a home in 5 years", help="Describe your financial goals and timeline")
        
        # Portfolio Details
        st.markdown('<div class="section-header"><h3>💼 Current Portfolio Details</h3></div>', unsafe_allow_html=True)
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            mutual_funds = st.text_area("📈 Mutual Funds", placeholder="Axis Bluechip - Equity - ₹2L\nHDFC Mid-Cap - ₹1.5L", help="List your mutual fund investments")
            stocks = st.text_area("📊 Stocks", placeholder="Infosys - 10 shares - ₹1500\nTCS - 5 shares - ₹2000", help="List your stock holdings")
        with col2_2:
            real_estate = st.text_area("🏠 Real Estate", placeholder="Residential Apartment - Mumbai - ₹10L\nCommercial Property - Delhi - ₹15L", help="List your real estate investments")
            fixed_deposit = st.text_input("🏦 Fixed Deposit (₹)", placeholder="500000", help="Total fixed deposit amount")
        
        # Submit button
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🚀 Generate AI Financial Report")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Quick Metrics")
    st.markdown("""
    <div class="metric-label">Portfolio Status</div>
    <div class="metric-value">Ready for Analysis</div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Agents")
    st.markdown("""
    - 📊 Portfolio Analyst
    - 📈 Growth Strategist  
    - 💎 Value Strategist
    - 💼 Financial Advisor
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# StateFlow for dynamic workflow management
class StateFlow:
    def __init__(self):
        self.current_state = "INIT"
        self.user_data = {}
        self.analysis_result = None
        self.strategy = None
        self.recommendations = None
    
    def set_user_data(self, data: Dict[str, Any]):
        self.user_data = data
        self.current_state = "USER_DATA_LOADED"
    
    def set_analysis_result(self, result: str):
        self.analysis_result = result
        self.current_state = "ANALYSIS_COMPLETE"
    
    def set_strategy(self, strategy: str):
        self.strategy = strategy
        self.current_state = "STRATEGY_DETERMINED"
    
    def set_recommendations(self, recommendations: str):
        self.recommendations = recommendations
        self.current_state = "RECOMMENDATIONS_READY"
    
    def get_next_agent(self):
        if self.current_state == "INIT":
            return "PortfolioAnalyst"
        elif self.current_state == "ANALYSIS_COMPLETE":
            return "GrowthStrategist" if self.strategy == "Growth" else "ValueStrategist"
        elif self.current_state == "RECOMMENDATIONS_READY":
            return "FinancialAdvisor"
        else:
            return None

# Initialize StateFlow
state_flow = StateFlow()

# Step 1: User Proxy Agent
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config=False
)

# Step 2: Portfolio Analysis Agent
portfolio_analyst = AssistantAgent(
    name="PortfolioAnalyst",
    llm_config={"config_list": config_list_gemini},
    system_message="""
    You are a Portfolio Analysis Agent. Analyze the user's portfolio and determine investment strategy.
    
    Your task:
    1. Analyze the user's current portfolio, salary, age, expenses, and risk tolerance
    2. Determine if they should pursue Growth or Value investment strategy
    3. Provide a brief explanation for your recommendation
    
    Output ONLY in JSON format: {"strategy": "Growth" or "Value", "reason": "brief explanation"}
    
    After analysis, mention "ANALYSIS_COMPLETE" to trigger next agent.
    """
)

# Step 3: Growth Investment Agent
growth_strategist = AssistantAgent(
    name="GrowthStrategist",
    llm_config={"config_list": config_list_gemini},
    system_message="""
    You are a Growth Investment Agent. Suggest high-growth investments for maximizing portfolio growth.
    
    Your task:
    1. Analyze the user's profile and current portfolio
    2. Suggest high-growth investment options (mid-cap mutual funds, global ETFs, tech stocks, etc.)
    3. Provide rationale for each recommendation
    
    Output: {"recommendations": ["item1", "item2", ...], "rationale": "brief explanation"}
    
    After recommendations, mention "RECOMMENDATIONS_READY" to trigger next agent.
    """
)

# Step 4: Value Investment Agent
value_strategist = AssistantAgent(
    name="ValueStrategist",
    llm_config={"config_list": config_list_gemini},
    system_message="""
    You are a Value Investment Agent. Suggest stable investments for long-term value.
    
    Your task:
    1. Analyze the user's profile and current portfolio
    2. Suggest stable, long-term investment options (bonds, blue-chip stocks, government schemes)
    3. Provide rationale for each recommendation
    
    Output: {"recommendations": ["item1", "item2", ...], "rationale": "brief explanation"}
    
    After recommendations, mention "RECOMMENDATIONS_READY" to trigger next agent.
    """
)

# Step 5: Investment Advisor Agent
financial_advisor = AssistantAgent(
    name="FinancialAdvisor",
    llm_config={"config_list": config_list_gemini},
    system_message="""
    You are an Investment Advisor Agent. Compile a comprehensive financial report.
    
    Your task:
    1. Review all previous agent outputs
    2. Generate a detailed, personalized financial report including:
       - Portfolio Analysis Summary
       - Recommended Strategy
       - Specific Investment Recommendations
       - Implementation Plan
       - Risk Assessment
    3. Format the report in Markdown with clear sections
    
    Add "TERMINATE" at the end when the report is complete.
    """
)

# Group Chat Manager for agent collaboration
def create_group_chat():
    # Create the group chat
    groupchat = GroupChat(
        agents=[user_proxy, portfolio_analyst, growth_strategist, value_strategist, financial_advisor],
        messages=[],
        max_round=50
    )
    
    # Create the group chat manager
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list_gemini}
    )
    
    return manager

def extract_strategy(content):
    try:
        # Find JSON in the content
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            data = json.loads(json_str)
            return data.get("strategy", "Growth")
    except:
        pass
    return "Growth"

def manage_investment_portfolio():
    # Prepare user data
    user_data = {
        "age": age,
        "salary": salary,
        "expenses": expenses,
        "goals": goals,
        "risk": risk,
        "mutual_funds": mutual_funds,
        "stocks": stocks,
        "real_estate": real_estate,
        "fixed_deposit": fixed_deposit
    }
    
    # Set user data in StateFlow
    state_flow.set_user_data(user_data)
    
    # Create the initial message
    initial_message = f"""
User Profile:
- Age: {age}
- Annual Salary: ₹{salary}
- Annual Expenses: ₹{expenses}
- Risk Tolerance: {risk}
- Financial Goals: {goals}

Current Portfolio:
- Mutual Funds: {mutual_funds or 'None'}
- Stocks: {stocks or 'None'}
- Real Estate: {real_estate or 'None'}
- Fixed Deposit: ₹{fixed_deposit or '0'}

Please analyze this portfolio and determine the investment strategy.
"""
    
    # Create group chat manager
    manager = create_group_chat()
    
    # Start the group chat with UserProxy initiating
    chat_result = user_proxy.initiate_chat(
        manager,
        message=initial_message,
        summary_method="last_msg",
        silent=True
    )
    
    # Extract the final report
    final_message = chat_result.chat_history[-1]["content"]
    if "TERMINATE" in final_message:
        return final_message.split("TERMINATE")[0].strip()
    return final_message

# ⏳ Generate and Display
if submit:
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update progress
    progress_bar.progress(25)
    status_text.markdown('<div class="status-indicator status-processing">🤖 Initializing AI Agents...</div>', unsafe_allow_html=True)
    
    try:
        # Generate report
        progress_bar.progress(50)
        status_text.markdown('<div class="status-indicator status-processing">📊 Analyzing Portfolio...</div>', unsafe_allow_html=True)
        
        result = manage_investment_portfolio()
        
        progress_bar.progress(100)
        status_text.markdown('<div class="status-indicator status-success">✅ Analysis Complete!</div>', unsafe_allow_html=True)
        
        # Display results
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>📊 Your Personalized Financial Report</h2></div>', unsafe_allow_html=True)
        st.markdown(result)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Workflow details
        with st.expander("🔍 AI Workflow Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🤖 Agent Collaboration**")
                st.markdown("""
                - Portfolio Analyst → Strategy Selection
                - Growth/Value Strategist → Recommendations
                - Financial Advisor → Final Report
                """)
            with col2:
                st.markdown("**📈 StateFlow Status**")
                st.markdown(f"""
                - Current State: {state_flow.current_state}
                - Strategy: {state_flow.strategy or 'Pending'}
                - Analysis: Complete ✅
                """)
            with col3:
                st.markdown("**⚡ Performance**")
                st.markdown("""
                - Multi-Agent Processing
                - Real-time Collaboration
                - Intelligent Routing
                """)
                
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        st.info("💡 Please check your inputs and try again. If the problem persists, try reducing the amount of text in your inputs.")