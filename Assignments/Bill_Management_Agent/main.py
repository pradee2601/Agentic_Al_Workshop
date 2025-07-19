import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import tempfile
import json
import google.generativeai as genai
from autogen.agentchat import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from autogen import config_list_from_json

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# --- MODERN UI CONFIG ---
st.set_page_config(
    page_title="🧾 Bill Management Agent", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS Styling - Dark Mode Compatible
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        min-height: 100vh;
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: white;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 1.2rem;
        opacity: 0.9;
        color: white;
    }
    
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
    }
    
    .category-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    
    .category-card:hover {
        transform: translateY(-5px);
    }
    
    .category-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: white;
    }
    
    .expense-item {
        background: rgba(255,255,255,0.15);
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
    }
    
    .summary-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .summary-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
        color: #ffffff;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .total-amount {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        color: #ff6b6b;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .highest-category {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        margin-top: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .chat-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .agent-message {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        color: #ffffff;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .manager-message {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        color: #ffffff;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .message-sender {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        opacity: 0.8;
        color: #ffffff;
    }
    
    .message-content {
        font-size: 1rem;
        line-height: 1.5;
        color: #ffffff;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        color: white;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.15);
    }
    
    .success-badge {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin: 1rem 0;
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin: 1rem 0;
    }
    
    .info-badge {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin: 1rem 0;
    }
    
    .spinner-container {
        text-align: center;
        padding: 2rem;
        color: white;
    }
    
    .error-container {
        background: rgba(255, 154, 158, 0.2);
        backdrop-filter: blur(10px);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid rgba(255, 154, 158, 0.3);
    }
    
    /* Make all text white for better visibility in dark mode */
    .stMarkdown, .stText, .stButton, .stSelectbox, .stTextInput {
        color: white !important;
    }
    
    /* Style file uploader for dark mode */
    .stFileUploader {
        color: white !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Modern Header
st.markdown("""
    <div class="main-header">
        <h1>💼 AI Bill Management Agent</h1>
        <p>Upload your bills and let AI categorize and analyze your expenses with intelligent insights</p>
    </div>
""", unsafe_allow_html=True)

# --- Upload File Section ---
st.markdown("""
    <div class="upload-section">
        <h3 style="font-family: 'Inter', sans-serif; font-weight: 600; color: #2c3e50; text-align: center; margin-bottom: 1.5rem;">
            📤 Upload Your Bill
        </h3>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

chat_log = []

# --- Gemini Vision to extract expense categories ---
def process_bill_with_gemini(image_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(image_file.read())
        tmp_path = tmp.name

    image = Image.open(tmp_path)

    response = model.generate_content([
        "Extract all expenses from this bill image. Group them into categories: Groceries, Dining, Utilities, Shopping, Entertainment, Others. Return as JSON format like {category: [{item, cost}]}.",
        image
    ])

    try:
        text = response.text.strip()
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        data = json.loads(text[json_start:json_end])
        return data, response.text
    except Exception as e:
        return None, response.text

# --- AutoGen Agents with Proper LLM Configuration ---
# Configure LLM for agents
config_list = [
    {
        "model": "gemini-1.5-flash",
        "api_key": GEMINI_API_KEY,
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
}

# User Proxy Agent
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    llm_config=llm_config,
    system_message="You are a user proxy that initiates conversations and provides bill images to the group manager."
)

# Bill Processing Agent
bill_processing_agent = AssistantAgent(
    name="BillProcessingAgent",
    llm_config=llm_config,
    system_message="""You are a bill processing agent that:
    1. Receives bill images and extracts expense data
    2. Categorizes expenses into: Groceries, Dining, Utilities, Shopping, Entertainment, Others
    3. Returns a structured list of categorized expenses with totals
    4. Always respond with clear categorization results"""
)

# Expense Summarization Agent
summary_agent = AssistantAgent(
    name="ExpenseSummarizationAgent",
    llm_config=llm_config,
    system_message="""You are an expense summarization agent that:
    1. Analyzes categorized expense data
    2. Calculates total spending per category
    3. Identifies the highest spending category
    4. Highlights unusual spending patterns
    5. Provides actionable insights on spending trends
    6. Always provide a comprehensive summary with totals and insights"""
)

# Group Chat Manager
group_chat = GroupChat(
    agents=[user_proxy, bill_processing_agent, summary_agent],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config
)

# --- Main Execution Flow ---
if uploaded_file:
    st.markdown('<div class="success-badge">✅ File uploaded successfully! Processing your bill...</div>', unsafe_allow_html=True)

    with st.spinner("🔍 Extracting expenses with AI..."):
        categorized_data, raw_response = process_bill_with_gemini(uploaded_file)

    if not categorized_data:
        st.markdown('<div class="error-container">❌ Failed to extract expenses from the image. Please try with a clearer image.</div>', unsafe_allow_html=True)
        st.text(raw_response)
    else:
        # Initialize chat log
        chat_log = []
        
        # Step 1: User Proxy initiates conversation with Group Manager
        st.markdown('<div class="info-badge">🤖 Starting AI agent collaboration...</div>', unsafe_allow_html=True)
        
        # User Proxy sends bill data to Group Manager
        user_message = f"I have uploaded a bill image. Here are the extracted expenses: {json.dumps(categorized_data, indent=2)}. Please process this data through the appropriate agents."
        chat_log.append(("UserProxy → Group Manager", user_message))
        
        # Step 2: Group Manager routes to Bill Processing Agent
        try:
            # Start the conversation
            user_proxy.send(user_message, manager)
            
            # Get the conversation history
            for message in group_chat.messages:
                sender = message.get("name", "Unknown")
                content = message.get("content", "")
                role = message.get("role", "")
                
                if role == "user":
                    chat_log.append(("UserProxy", content))
                elif role == "assistant":
                    if "BillProcessingAgent" in sender:
                        chat_log.append(("BillProcessingAgent", content))
                    elif "ExpenseSummarizationAgent" in sender:
                        chat_log.append(("ExpenseSummarizationAgent", content))
                    elif "Group Manager" in sender or "manager" in sender.lower():
                        chat_log.append(("Group Manager", content))
            
        except Exception as e:
            st.markdown(f'<div class="error-container">Error in agent collaboration: {str(e)}</div>', unsafe_allow_html=True)
            # Fallback to manual processing
            chat_log.append(("BillProcessingAgent", "Processing bill data and categorizing expenses..."))
            chat_log.append(("ExpenseSummarizationAgent", "Analyzing spending patterns and generating summary..."))

        # --- Display Categorized Expenses ---
        st.markdown("## 📂 Categorized Expenses")
        
        # Category icons mapping
        category_icons = {
            "Groceries": "🛒",
            "Dining": "🍽️",
            "Utilities": "⚡",
            "Shopping": "🛍️",
            "Entertainment": "🎬",
            "Others": "📦"
        }
        
        for category, items in categorized_data.items():
            if items:
                icon = category_icons.get(category, "📦")
                st.markdown(f"""
                    <div class="category-card">
                        <div class="category-title">
                            {icon} {category}
                        </div>
                """, unsafe_allow_html=True)
                
                total_category = sum(float(i['cost']) for i in items)
                for i in items:
                    st.markdown(f"""
                        <div class="expense-item">
                            <strong>{i['item']}</strong>: ₹{i['cost']}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                        <div style="text-align: right; margin-top: 1rem; font-weight: 600; font-size: 1.1rem;">
                            Category Total: ₹{total_category:.2f}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Calculate overall summary
        total_expenditure = sum(
            sum(float(i['cost']) for i in items) 
            for items in categorized_data.values() 
            if items
        )
        
        # --- Spending Summary Section ---
        st.markdown("""
            <div class="summary-card">
                <div class="summary-title">📋 Spending Summary</div>
                <div class="total-amount">₹{:.2f}</div>
        """.format(total_expenditure), unsafe_allow_html=True)
        
        # Find highest spending category
        category_totals = {}
        for category, items in categorized_data.items():
            if items:
                category_totals[category] = sum(float(i['cost']) for i in items)
        
        if category_totals:
            highest_category = max(category_totals, key=category_totals.get)
            highest_amount = category_totals[highest_category]
            
            st.markdown(f"""
                <div class="highest-category">
                    <strong>Highest Spending Category:</strong> {highest_category} (₹{highest_amount:.2f})
                </div>
            """, unsafe_allow_html=True)
            
            # Generate insights
            if highest_amount > total_expenditure * 0.5:
                st.markdown('<div class="warning-badge">⚠️ High spending alert: This category represents more than 50% of total expenses.</div>', unsafe_allow_html=True)
            elif highest_amount > total_expenditure * 0.3:
                st.markdown('<div class="info-badge">ℹ️ Moderate spending: This category represents a significant portion of expenses.</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Agent Chat Logs ---
        st.markdown("""
            <div class="chat-container">
                <div class="chat-title">💬 AI Agent Collaboration Logs</div>
        """, unsafe_allow_html=True)
        
        for sender, message in chat_log:
            if "UserProxy" in sender:
                style = "user-message"
            elif "Group Manager" in sender:
                style = "manager-message"
            else:
                style = "agent-message"
            
            st.markdown(f"""
                <div class="{style}">
                    <div class="message-sender">{sender}</div>
                    <div class="message-content">{message}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)