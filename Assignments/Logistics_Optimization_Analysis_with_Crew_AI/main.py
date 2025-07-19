# -*- coding: utf-8 -*-
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Set page configuration
st.set_page_config(page_title="Logistics Optimizer", layout="wide", page_icon="🚚")

# Add CSS for button/input alignment and card look using theme variables
st.markdown(
    """
    <style>
    .themed-card {
        background: var(--background-color);
        border-radius: 12px;
        padding: 2.5rem 2rem 1.5rem 2rem;
        box-shadow: 0 2px 8px rgba(79,139,249,0.07);
        margin-bottom: 2rem;
        border: 1px solid var(--secondary-background-color);
    }
    .themed-result {
        background: var(--secondary-background-color);
        border-radius: 12px;
        padding: 2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(79,139,249,0.10);
        margin-top: 2rem;
    }
    .stTextInput > div > div > input, .stButton > button {
        min-height: 48px;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>🚚 Logistics Optimizer</h2>
            <p style='font-size: 16px;'>AI-powered logistics analysis and optimization using CrewAI</p>
            <hr style='border: 1px solid #eee;'>
            <p style='font-size: 14px; color: #888;'>Developed for logistics industry problem-solving.<br>Powered by Gemini & CrewAI.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Title
st.markdown(
    """
    <h1 style='font-size: 2.8rem; color: var(--text-color); margin-bottom: 0;'>🚚 Logistics Optimization <span style='color:#4F8BF9;'>AI</span></h1>
    <p style='font-size: 1.2rem; color: var(--text-color); margin-top: 0;'>Analyze and optimize delivery routes and inventory management for your products.</p>
    <hr style='border: 1px solid var(--secondary-background-color);'>
    """,
    unsafe_allow_html=True
)

# Card-like input form using theme variables and a form for alignment
st.markdown('<div class="themed-card">', unsafe_allow_html=True)
with st.form("logistics_form"):
    cols = st.columns([3, 1])
    with cols[0]:
        product_input = st.text_input("Enter product names separated by commas", "TV, Laptops, Headphones")
    with cols[1]:
        submitted = st.form_submit_button("🚀 Optimize Logistics")
st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    with st.spinner("<span style='color:#4F8BF9;'>Running CrewAI agents for logistics optimization...</span>"):
        # Prepare LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # Define tools (empty for now)
        def logistics_analyst_tools():
            return []

        def optimization_strategist_tools():
            return []

        # Define agents
        logistics_analyst = Agent(
            role="Logistics Analyst",
            goal="Analyze logistics operations to find inefficiencies in delivery routes and inventory turnover.",
            backstory="A seasoned analyst with years of experience in identifying bottlenecks in supply chain networks.",
            verbose=True,
            llm=llm,
            tools=logistics_analyst_tools()
        )

        optimization_strategist = Agent(
            role="Optimization Strategist",
            goal="Design data-driven strategies to optimize logistics operations and improve performance.",
            backstory="Known for implementing cost-saving logistics strategies using advanced AI models.",
            verbose=True,
            llm=llm,
            tools=optimization_strategist_tools()
        )

        # Parse product input
        products = [p.strip() for p in product_input.split(",") if p.strip()]

        # Define tasks
        task1 = Task(
            description=f"Analyze logistics data for the following products: {products}. Focus on delivery routes and inventory turnover trends.",
            expected_output="Summary of current inefficiencies and potential improvement areas in logistics operations.",
            agent=logistics_analyst
        )

        task2 = Task(
            description="Based on the logistics analyst's findings, develop an optimization strategy to reduce delivery time and improve inventory management.",
            expected_output="Detailed optimization strategy with action points to improve logistics efficiency.",
            agent=optimization_strategist
        )

        # Create Crew
        crew = Crew(
            agents=[logistics_analyst, optimization_strategist],
            tasks=[task1, task2],
            verbose=True
        )

        # Execute CrewAI workflow
        result = crew.kickoff()

    # Show result in a modern card using theme variables
    st.markdown(
        '<div class="themed-result">\n'
        '<h3 style="color:#4F8BF9; margin-bottom: 0.5rem;">&#x2705; Optimization Complete!</h3>'
        '<h4 style="color:var(--text-color); margin-top: 0;">&#x1F50D; Final Optimization Strategy</h4>',
        unsafe_allow_html=True
    )
    st.markdown(result)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr style='border: 1px solid var(--secondary-background-color); margin-top: 3rem;'>
    <div style='text-align: center; color: var(--text-color); font-size: 0.95rem;'>
        &copy; 2024 Logistics Optimizer &mdash; Powered by CrewAI, Gemini, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)