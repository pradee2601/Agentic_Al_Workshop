import streamlit as st
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import pandas as pd
import plotly.express as px
import json
import base64
from datetime import datetime

# Load environment variables
load_dotenv()

class Competitor(BaseModel):
    name: str = Field(description="Name of the competitor")
    website: str = Field(description="Website URL of the competitor")
    description: str = Field(description="One-line description of the competitor")
    features: List[str] = Field(description="List of features offered")
    pricing_model: str = Field(description="Pricing tier or model")
    target_audience: str = Field(description="Target audience description")
    usp: str = Field(description="Unique selling proposition")
    market_share: str = Field(description="Estimated market share or position")
    funding_status: str = Field(description="Latest funding round and amount")
    user_rating: str = Field(description="Average user rating if available")
    feature_categories: Dict[str, List[str]] = Field(description="Features organized by category")

class DifferentiationStrategy(BaseModel):
    underserved_niches: List[str] = Field(description="List of underserved market niches")
    tech_innovations: List[str] = Field(description="Potential technological innovations")
    pricing_disruptions: List[str] = Field(description="Potential pricing model disruptions")
    competitive_advantages: List[str] = Field(description="Potential competitive advantages")

def initialize_models():
    """Initialize the AI models."""
    try:
        # Initialize Tavily search
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            st.error("Please set TAVILY_API_KEY in your .env file")
            return None, None
            
        search = TavilySearch(
            api_key=tavily_api_key,
            max_results=10,
            search_depth="advanced"
        )
        
        # Initialize Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("Please set GOOGLE_API_KEY in your .env file")
            return None, None
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )
        
        return search, llm
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None

def analyze_competitors(startup_idea: str, search, llm) -> tuple[List[Dict], Dict]:
    """Analyze competitors and generate differentiation strategy."""
    
    # Search for competitors
    search_query = f"top competitors and alternatives for {startup_idea} in tech industry, include company names, websites, and features"
    search_results = search.invoke({"query": search_query})
    
    # Process search results
    competitors = []
    seen_names = set()
    
    if isinstance(search_results, dict) and "results" in search_results:
        results = search_results["results"]
    else:
        results = search_results
    
    # Extract competitor information
    for result in results:
        if isinstance(result, dict):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            
            # Extract competitor names
            competitor_names = extract_competitor_names(title, content)
            
            for name in competitor_names:
                if name and name not in seen_names:
                    seen_names.add(name)
                    competitor = {
                        "name": name,
                        "website": url if url else "N/A",
                        "description": content[:200] + "..." if content else "No description available",
                        "features": [],  # Will be filled by Gemini
                        "pricing_model": "N/A",  # Will be filled by Gemini
                        "target_audience": "N/A",  # Will be filled by Gemini
                        "usp": "N/A",  # Will be filled by Gemini
                        "market_share": "N/A",  # Will be filled by Gemini
                        "funding_status": "N/A",  # Will be filled by Gemini
                        "user_rating": "N/A",  # Will be filled by Gemini
                        "feature_categories": {}  # Will be filled by Gemini
                    }
                    competitors.append(competitor)
    
    # Use Gemini to enrich competitor information
    if competitors and llm:
        enriched_competitors = enrich_competitor_data(competitors, startup_idea, llm)
        # Validate enriched data structure
        validated_competitors = []
        for comp in enriched_competitors:
            if isinstance(comp, dict):
                validated_comp = {
                    "name": comp.get("name", "Unknown"),
                    "website": comp.get("website", "N/A"),
                    "description": comp.get("description", "No description available"),
                    "features": comp.get("features", []),
                    "pricing_model": comp.get("pricing_model", "N/A"),
                    "target_audience": comp.get("target_audience", "N/A"),
                    "usp": comp.get("usp", "N/A"),
                    "market_share": comp.get("market_share", "N/A"),
                    "funding_status": comp.get("funding_status", "N/A"),
                    "user_rating": comp.get("user_rating", "N/A"),
                    "feature_categories": comp.get("feature_categories", {})
                }
                # Ensure features is a list
                if not isinstance(validated_comp["features"], list):
                    validated_comp["features"] = []
                validated_competitors.append(validated_comp)
        
        differentiation_strategy = generate_differentiation_strategy(startup_idea, validated_competitors, llm)
        return validated_competitors, differentiation_strategy
    
    return competitors, {}

def extract_competitor_names(title: str, content: str) -> List[str]:
    """Extract competitor names from title and content."""
    names = []
    
    # Extract from title
    if "competitor" in title.lower() or "alternative" in title.lower():
        parts = title.lower().split("of")
        if len(parts) > 1:
            names.append(parts[0].strip())
    
    # Extract from content
    if "competitors include" in content.lower():
        parts = content.lower().split("competitors include")
        if len(parts) > 1:
            names.extend([name.strip() for name in parts[1].split(",") if name.strip()])
    
    return list(set(names))  # Remove duplicates

def enrich_competitor_data(competitors: List[Dict], startup_idea: str, llm) -> List[Dict]:
    """Use Gemini to enrich competitor information."""
    # Filter out non-company entries (like comparison sites, directories)
    filtered_competitors = []
    for comp in competitors:
        if comp["name"] and not any(site in comp["website"].lower() for site in ["capterra", "softwareworld", "alternatives", "comparison"]):
            filtered_competitors.append(comp)
    
    if not filtered_competitors:
        return competitors

    # First pass: Get initial data
    initial_prompt = f"""You are a business analyst specializing in auto repair shop management software analysis. For each competitor listed below, provide detailed information in a structured JSON format.

Competitors to analyze:
{json.dumps(filtered_competitors, indent=2)}

For each competitor, provide the following information in a JSON array format:
1. name: Keep the original name
2. website: The company's website URL
3. description: A 2-3 sentence description of what the company does, focusing on their core business
4. features: An array of 7-10 key features or capabilities that are specific to this company, focusing on:
   - Appointment scheduling and management
   - Customer management and communication
   - Service history tracking
   - Parts inventory management
   - Digital vehicle inspections
   - Payment processing
   - Reporting and analytics
   - Integration capabilities
   - Mobile app features
   - Customer portal features
5. pricing_model: The specific pricing structure or model
6. target_audience: Specific target audience segments (e.g., "Small automotive repair shops with 1-5 bays")
7. usp: The unique selling proposition or main differentiator from competitors
8. market_share: Estimated market share or position in the market
9. funding_status: Latest funding round and amount if available
10. user_rating: Average user rating if available
11. feature_categories: Organize features into categories like:
    - Core Management Features
    - Customer Experience Features
    - Inventory & Parts Management
    - Financial & Payment Features
    - Reporting & Analytics
    - Integration & API Features
    - Mobile & Remote Access

Important guidelines:
- Only include real, specific information. If information is not available, use "Information not available" instead of "N/A"
- Focus on actual companies, not comparison sites or directories
- Ensure features are specific to each company, not generic industry features
- Make sure pricing information is specific when available
- Target audience should be specific and detailed
- USP should highlight what makes this company unique
- For features, focus on concrete capabilities rather than marketing language

Please analyze each competitor and return the data in valid JSON format."""
    
    try:
        # Get initial data
        response = llm.invoke(initial_prompt)
        response_text = response.content
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            enriched_data = json.loads(json_str)
            
            # Second pass: Generate missing information
            for comp in enriched_data:
                missing_fields = []
                if comp.get("features") in [None, [], ["Information not available"]]:
                    missing_fields.append("features")
                if comp.get("pricing_model") == "Information not available":
                    missing_fields.append("pricing_model")
                if comp.get("target_audience") == "Information not available":
                    missing_fields.append("target_audience")
                if comp.get("usp") == "Information not available":
                    missing_fields.append("usp")
                
                if missing_fields:
                    generate_prompt = f"""Based on the following information about {comp['name']}, generate realistic and plausible information for the missing fields: {', '.join(missing_fields)}.

Company Information:
{json.dumps(comp, indent=2)}

Startup Idea: {startup_idea}

Please generate information that would be realistic for an auto repair shop management software company. Return the response in JSON format with only the missing fields.
Example format:
{{
    "features": [
        "Digital vehicle inspection forms",
        "Automated appointment scheduling",
        "Parts inventory tracking",
        "Customer communication portal",
        "Service history database",
        "Payment processing integration",
        "Mobile technician app"
    ],
    "pricing_model": "Starting at $199/month for small shops, enterprise pricing available",
    "target_audience": "Independent auto repair shops with 2-10 bays",
    "usp": "All-in-one solution with best-in-class digital inspection tools"
}}"""
                    
                    try:
                        gen_response = llm.invoke(generate_prompt)
                        gen_text = gen_response.content
                        gen_start = gen_text.find('{')
                        gen_end = gen_text.rfind('}') + 1
                        if gen_start != -1 and gen_end != -1:
                            gen_json = json.loads(gen_text[gen_start:gen_end])
                            for field in missing_fields:
                                if field in gen_json:
                                    comp[field] = gen_json[field]
                    except Exception as e:
                        st.warning(f"Could not generate missing information for {comp['name']}: {str(e)}")
            
            # Validate and clean the enriched data
            validated_data = []
            for comp in enriched_data:
                if isinstance(comp, dict):
                    validated_comp = {
                        "name": comp.get("name", "Unknown"),
                        "website": comp.get("website", "Information not available"),
                        "description": comp.get("description", "Information not available"),
                        "features": comp.get("features", ["Information not available"]),
                        "pricing_model": comp.get("pricing_model", "Information not available"),
                        "target_audience": comp.get("target_audience", "Information not available"),
                        "usp": comp.get("usp", "Information not available"),
                        "market_share": comp.get("market_share", "Information not available"),
                        "funding_status": comp.get("funding_status", "Information not available"),
                        "user_rating": comp.get("user_rating", "Information not available"),
                        "feature_categories": comp.get("feature_categories", {})
                    }
                    # Ensure features is a list and not empty
                    if not isinstance(validated_comp["features"], list) or not validated_comp["features"]:
                        validated_comp["features"] = ["Information not available"]
                    validated_data.append(validated_comp)
            return validated_data
        else:
            st.error("Could not find valid JSON in the response")
            return competitors
    except Exception as e:
        st.error(f"Error enriching competitor data: {str(e)}")
        return competitors

def generate_differentiation_strategy(startup_idea: str, competitors: List[Dict], llm) -> Dict:
    """Generate differentiation strategy using Gemini."""
    prompt = f"""You are a business strategy consultant. Based on the startup idea '{startup_idea}' and these competitors:
{json.dumps(competitors, indent=2)}

Generate a detailed differentiation strategy in the following JSON format:
{{
    "underserved_niches": ["List of 3-5 underserved market segments"],
    "tech_innovations": ["List of 3-5 potential technological innovations"],
    "pricing_disruptions": ["List of 3-5 potential pricing model innovations"],
    "competitive_advantages": ["List of 3-5 potential competitive advantages"]
}}

Make each list item specific and actionable. Focus on unique opportunities that aren't currently addressed by the competitors.
Return the response in valid JSON format."""
    
    try:
        response = llm.invoke(prompt)
        # Extract JSON from the response text
        response_text = response.content
        # Find the first occurrence of '{' and last occurrence of '}'
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            strategy = json.loads(json_str)
            
            # Validate and clean the strategy data
            validated_strategy = {
                "underserved_niches": strategy.get("underserved_niches", []),
                "tech_innovations": strategy.get("tech_innovations", []),
                "pricing_disruptions": strategy.get("pricing_disruptions", []),
                "competitive_advantages": strategy.get("competitive_advantages", [])
            }
            return validated_strategy
        else:
            st.error("Could not find valid JSON in the response")
            return {}
    except Exception as e:
        st.error(f"Error generating differentiation strategy: {str(e)}")
        return {}

def create_visualization(competitors: List[Dict]):
    """Create visualizations for the competitor analysis."""
    if not competitors:
        return None, None
        
    # Convert competitors to DataFrame
    competitors_df = pd.DataFrame(competitors)
    
    # Create feature comparison heatmap
    features = []
    for comp in competitors:
        if isinstance(comp, dict) and "features" in comp and isinstance(comp["features"], list):
            for feature in comp["features"]:
                features.append({
                    "Competitor": comp.get("name", "Unknown"),
                    "Feature": feature,
                    "Has Feature": True
                })
    
    features_df = pd.DataFrame(features)
    if not features_df.empty:
        pivot_df = features_df.pivot_table(
            index="Competitor",
            columns="Feature",
            values="Has Feature",
            fill_value=False
        )
        
        # Create heatmap
        fig = px.imshow(
            pivot_df,
            title="Feature Comparison Matrix",
            labels=dict(x="Features", y="Competitors", color="Has Feature"),
            color_continuous_scale=["white", "blue"]
        )
        
        # Create feature category comparison
        category_data = []
        for comp in competitors:
            if isinstance(comp, dict) and "feature_categories" in comp:
                for category, features in comp["feature_categories"].items():
                    category_data.append({
                        "Competitor": comp.get("name", "Unknown"),
                        "Category": category,
                        "Feature Count": len(features)
                    })
        
        if category_data:
            category_df = pd.DataFrame(category_data)
            category_fig = px.bar(
                category_df,
                x="Competitor",
                y="Feature Count",
                color="Category",
                title="Feature Categories Comparison",
                barmode="group"
            )
            return fig, category_fig
            
        return fig, None
    return None, None

# Add caching decorator
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_analysis(startup_idea: str, _search, _llm):
    return analyze_competitors(startup_idea, _search, _llm)

def create_export_data(competitors: List[Dict], strategy: Dict) -> Dict:
    """Create a structured export of the analysis."""
    return {
        "timestamp": datetime.now().isoformat(),
        "competitors": competitors,
        "differentiation_strategy": strategy
    }

def get_download_link(data: Dict, filename: str) -> str:
    """Generate a download link for the analysis data."""
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}">Download Analysis</a>'

def main():
    st.set_page_config(
        page_title="Competitor Analysis & Differentiation Strategy",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Competitor Analysis & Differentiation Strategy")
    st.markdown("""
    This tool helps founders position their startup in crowded markets by:
    - Identifying top competitors
    - Analyzing features, pricing, and target audiences
    - Generating differentiation strategies
    - Visualizing feature gaps
    """)
    
    # Initialize models
    search, llm = initialize_models()
    
    if search is None or llm is None:
        st.error("Please check your API keys in the .env file")
        return
    
    # Get startup idea from user
    startup_idea = st.text_input("Enter your startup idea:", placeholder="e.g., AI-powered customer service platform")
    
    if st.button("Analyze Competitors"):
        if not startup_idea:
            st.warning("Please enter a startup idea")
            return
            
        with st.spinner("Analyzing competitors and generating strategy..."):
            try:
                # Use cached analysis
                competitors, strategy = get_cached_analysis(startup_idea, search, llm)
                
                if not competitors:
                    st.warning("No competitors found. Try a different search term.")
                    return
                
                # Create export data
                export_data = create_export_data(competitors, strategy)
                export_filename = f"competitor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Competitors", "Differentiation Strategy", "Feature Analysis", "Export"])
                
                with tab1:
                    for comp in competitors:
                        with st.expander(f"📊 {comp['name']}"):
                            st.markdown(f"**Website:** {comp['website']}")
                            st.markdown(f"**Description:** {comp['description']}")
                            st.markdown("**Features:**")
                            for feature in comp['features']:
                                st.markdown(f"- {feature}")
                            st.markdown(f"**Pricing Model:** {comp['pricing_model']}")
                            st.markdown(f"**Target Audience:** {comp['target_audience']}")
                            st.markdown(f"**USP:** {comp['usp']}")
                
                with tab2:
                    st.markdown("### 🎯 Differentiation Strategy")
                    if strategy:
                        st.markdown("#### Underserved Niches")
                        for niche in strategy.get("underserved_niches", []):
                            st.markdown(f"- {niche}")
                        
                        st.markdown("#### Potential Tech Innovations")
                        for innovation in strategy.get("tech_innovations", []):
                            st.markdown(f"- {innovation}")
                        
                        st.markdown("#### Pricing Disruptions")
                        for disruption in strategy.get("pricing_disruptions", []):
                            st.markdown(f"- {disruption}")
                        
                        st.markdown("#### Competitive Advantages")
                        for advantage in strategy.get("competitive_advantages", []):
                            st.markdown(f"- {advantage}")
                    else:
                        st.warning("Could not generate differentiation strategy.")
                
                with tab3:
                    st.markdown("### 📊 Feature Analysis")
                    fig, category_fig = create_visualization(competitors)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        if category_fig:
                            st.plotly_chart(category_fig, use_container_width=True)
                    else:
                        st.info("No features found to display in the comparison matrix.")
                
                with tab4:
                    st.markdown("### 📥 Export Analysis")
                    st.markdown("Download the complete analysis in JSON format:")
                    st.markdown(get_download_link(export_data, export_filename), unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 