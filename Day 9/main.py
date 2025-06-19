import streamlit as st
import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime
import json
import base64

from market_analysis_graph import run_market_analysis

# Load environment variables
load_dotenv()

def initialize_models():
    """Initialize the AI models."""
    try:
        # Initialize Tavily search
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            st.error("Please set TAVILY_API_KEY in your .env file")
            return None, None, None
            
        search = TavilySearch(
            api_key=tavily_api_key,
            max_results=10,
            search_depth="advanced"
        )
        
        # Initialize Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("Please set GOOGLE_API_KEY in your .env file")
            return None, None, None
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )

        # Initialize vector database
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        return search, llm, vector_db
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None, None

def create_export_data(competitors: list, strategy: dict, feature_matrix: dict, visualizations: dict) -> dict:
    """Create a structured export of the analysis."""
    return {
        "timestamp": datetime.now().isoformat(),
        "competitors": competitors,
        "differentiation_strategy": strategy,
        "feature_matrix": feature_matrix,
        "visualizations": visualizations
    }

def get_download_link(data: dict, filename: str) -> str:
    """Generate a download link for the analysis data."""
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}">Download Analysis</a>'

def main():
    st.set_page_config(
        page_title="Competitor Differentiation Mapper",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Competitor Differentiation Mapper")
    st.markdown("""
    This tool helps founders craft clear market positioning by:
    - Discovering and analyzing competitors
    - Building comprehensive feature matrices
    - Identifying differentiation opportunities
    - Generating strategic positioning recommendations
    - Visualizing market gaps and opportunities
    """)
    
    # Initialize models
    search, llm, vector_db = initialize_models()
    
    if search is None or llm is None or vector_db is None:
        st.error("Please check your API keys in the .env file")
        return
    
    # Startup Idea Input
    st.header("💡 Startup Idea")
    startup_idea = st.text_area(
        "Describe your startup idea:",
        placeholder="Describe your startup idea in detail, including the problem you're solving and your target market..."
    )
    
    if st.button("Analyze Market"):
        if not startup_idea:
            st.warning("Please enter your startup idea")
            return
            
        with st.spinner("Analyzing market and competitors..."):
            try:
                # Run the market analysis workflow
                final_state = run_market_analysis(startup_idea, search, llm, vector_db)
                
                if final_state.get("error"):
                    st.error(final_state["error"])
                    return
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Display competitors
                st.subheader("🏢 Competitors")
                for comp in final_state["competitors"]:
                    with st.expander(f"{comp['name']}"):
                        st.markdown(f"**Website:** {comp['website']}")
                        st.markdown(f"**Description:** {comp['description']}")
                        st.markdown("**Features:**")
                        for feature in comp['features']:
                            st.markdown(f"- {feature}")
                        st.markdown(f"**Pricing Model:** {comp['pricing_model']}")
                        st.markdown(f"**Target Audience:** {comp['target_audience']}")
                        st.markdown(f"**USP:** {comp['usp']}")
                
                # Display differentiation strategy
                st.subheader("💡 Differentiation Strategy")
                if final_state["strategy"]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Market Opportunities")
                        st.markdown("##### Whitespace Opportunities")
                        for opp in final_state["strategy"].get("whitespace_opportunities", []):
                            st.markdown(f"- {opp['description']} ({opp['type']})")
                        
                        st.markdown("##### Innovation Areas")
                        for opp in final_state["strategy"].get("innovation_areas", []):
                            st.markdown(f"- {opp['opportunity']} ({opp['category']})")
                    
                    with col2:
                        st.markdown("#### Strategic Positioning")
                        positioning = final_state["strategy"].get("positioning_strategy", {})
                        st.markdown(f"**Market Position:** {positioning.get('market_position', 'N/A')}")
                        st.markdown(f"**Value Proposition:** {positioning.get('value_proposition', 'N/A')}")
                        st.markdown(f"**Target Audience:** {positioning.get('target_audience', 'N/A')}")
                        st.markdown(f"**Brand Positioning:** {positioning.get('brand_positioning', 'N/A')}")
                        
                        st.markdown("##### Key Differentiators")
                        for diff in positioning.get("key_differentiators", []):
                            st.markdown(f"- {diff}")
                
                # Display visualizations
                st.subheader("📈 Market Analysis Visualizations")
                
                # Feature Gap Map
                st.markdown("##### Feature Gap Analysis")
                feature_gap_map = final_state["visualizations"].get("feature_gap_map", {})
                if feature_gap_map:
                    st.write(feature_gap_map)
                
                # Market Opportunity Map
                st.markdown("##### Market Opportunity Map")
                opportunity_map = final_state["visualizations"].get("market_opportunity_map", {})
                if opportunity_map:
                    st.write(opportunity_map)
                
                # Competitive Landscape
                st.markdown("##### Competitive Landscape")
                landscape = final_state["visualizations"].get("competitive_landscape", {})
                if landscape:
                    st.write(landscape)
                
                # Innovation Radar
                st.markdown("##### Innovation Radar")
                radar = final_state["visualizations"].get("innovation_radar", {})
                if radar:
                    st.write(radar)
                
                # Export functionality
                st.header("📥 Export Analysis")
                export_data = create_export_data(
                    final_state["competitors"],
                    final_state["strategy"],
                    final_state["feature_matrix"],
                    final_state["visualizations"]
                )
                export_filename = f"market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.markdown(get_download_link(export_data, export_filename), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 