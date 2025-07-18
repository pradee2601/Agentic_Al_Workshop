import streamlit as st
from typing import List, Dict
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from typing_extensions import TypedDict

# --- Tool Definitions ---
FEATURE_CATEGORIES = [
    "Core Features",
    "User Experience",
    "Technical Capabilities",
    "Integration Features",
    "Security & Privacy",
    "Analytics & Reporting",
    "Mobile & Remote Access"
]

class CategorizeFeaturesInput(TypedDict):
    features: list

@tool("categorize_features")
def categorize_features_tool(input: CategorizeFeaturesInput, llm=None) -> dict:
    """Categorizes features into predefined categories using an LLM."""
    features = input["features"]
    categorized = {category: [] for category in FEATURE_CATEGORIES}
    prompt = f"""Categorize the following features into these categories: {', '.join(FEATURE_CATEGORIES)}\n\nFeatures to categorize:\n{json.dumps(features, indent=2)}\n\nReturn the response in JSON format with categories as keys and lists of features as values."""
    try:
        response = llm.invoke(prompt)
        response_text = response.content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            categories = json.loads(response_text[start_idx:end_idx])
            for category, feats in categories.items():
                if category in categorized:
                    categorized[category] = feats
    except Exception as e:
        st.warning(f"Error categorizing features: {str(e)}")
    return categorized

class GenerateFeatureComparisonInput(TypedDict):
    competitors: list
    categorized_features: dict

@tool("generate_feature_comparison")
def generate_feature_comparison_tool(input: GenerateFeatureComparisonInput, llm=None) -> dict:
    """Generates a detailed feature comparison matrix for competitors and categorized features."""
    competitors = input["competitors"]
    categorized_features = input["categorized_features"]
    comparison = {
        "by_category": {},
        "by_competitor": {}
    }
    try:
        # Build comparison by category
        for category, features in categorized_features.items():
            comparison["by_category"][category] = {}
            for feature in features:
                comparison["by_category"][category][feature] = []
                for comp in competitors:
                    if isinstance(comp, dict) and feature in comp.get("features", []):
                        comparison["by_category"][category][feature].append(comp["name"])
        # Build comparison by competitor
        for comp in competitors:
            if isinstance(comp, dict):
                comparison["by_competitor"][comp["name"]] = {}
                for category, features in categorized_features.items():
                    comparison["by_competitor"][comp["name"]][category] = []
                    for feature in features:
                        if feature in comp.get("features", []):
                            comparison["by_competitor"][comp["name"]][category].append(feature)
    except Exception as e:
        st.warning(f"Error generating feature comparison: {str(e)}")
    return comparison

class FeatureMatrixBuilderAgent:
    """Agent responsible for building and analyzing feature matrices using LangChain agentic pattern."""
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.tools = [
            categorize_features_tool.bind(llm=llm),
            generate_feature_comparison_tool.bind(llm=llm)
        ]
        self.agent = initialize_agent(
            self.tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True
        )

    def build_feature_matrix(self, competitors: List[Dict]) -> Dict:
        """Build a comprehensive feature matrix from competitor data using the agent."""
        matrix = {
            "features": {},
            "pricing_comparison": {},
            "audience_segments": {},
            "usps": {}
        }
        try:
            # Extract all features and metadata
            all_features = set()
            for comp in competitors:
                if isinstance(comp, dict):
                    features = comp.get("features", [])
                    all_features.update(features)
                    matrix["pricing_comparison"][comp["name"]] = comp.get("pricing_model", "N/A")
                    matrix["audience_segments"][comp["name"]] = comp.get("target_audience", "N/A")
                    matrix["usps"][comp["name"]] = comp.get("usp", "N/A")
            # Use agent to categorize features
            categorized_features = self.agent.run({"features": list(all_features)}, tool_name="categorize_features")
            if isinstance(categorized_features, str):
                categorized_features = json.loads(categorized_features)
            matrix["features"] = categorized_features
            # Use agent to generate feature comparison
            feature_comparison = self.agent.run({
                "competitors": competitors,
                "categorized_features": categorized_features
            }, tool_name="generate_feature_comparison")
            if isinstance(feature_comparison, str):
                feature_comparison = json.loads(feature_comparison)
            matrix["feature_comparison"] = feature_comparison
            return matrix
        except Exception as e:
            st.error(f"Error building feature matrix: {str(e)}")
            return matrix

    def generate_visualization_data(self, matrix: Dict) -> Dict:
        """Generate data for visualization of the feature matrix."""
        viz_data = {
            "feature_heatmap": [],
            "category_comparison": [],
            "pricing_comparison": [],
            "audience_overlap": []
        }
        try:
            # Generate feature heatmap data
            for category, features in matrix["features"].items():
                for feature in features:
                    for comp_name, comp_features in matrix["feature_comparison"]["by_competitor"].items():
                        has_feature = feature in comp_features.get(category, [])
                        viz_data["feature_heatmap"].append({
                            "category": category,
                            "feature": feature,
                            "competitor": comp_name,
                            "has_feature": has_feature
                        })
            # Generate category comparison data
            for comp_name, categories in matrix["feature_comparison"]["by_competitor"].items():
                for category, features in categories.items():
                    viz_data["category_comparison"].append({
                        "competitor": comp_name,
                        "category": category,
                        "feature_count": len(features)
                    })
            # Generate pricing comparison data
            for comp_name, pricing in matrix["pricing_comparison"].items():
                viz_data["pricing_comparison"].append({
                    "competitor": comp_name,
                    "pricing": pricing
                })
        except Exception as e:
            st.warning(f"Error generating visualization data: {str(e)}")
        return viz_data 