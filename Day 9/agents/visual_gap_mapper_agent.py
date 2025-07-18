import streamlit as st
from typing import List, Dict
import re
import pandas as pd
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
import json
from typing_extensions import TypedDict


# --- Tool Definitions ---
class FeatureGapMapInput(TypedDict):
    competitors: list
    feature_matrix: dict

@tool("create_feature_gap_map")
def create_feature_gap_map_tool(input: FeatureGapMapInput) -> dict:
    """Creates an interactive feature gap map from competitors and feature matrix."""
    competitors = input["competitors"]
    feature_matrix = input["feature_matrix"]
    gap_map = {
        "type": "feature_gap_map",
        "data": {
            "categories": [],
            "features": [],
            "gaps": []
        }
    }
    try:
        for category, features in feature_matrix["features"].items():
            gap_map["data"]["categories"].append(category)
            for feature in features:
                gap_map["data"]["features"].append({
                    "name": feature,
                    "category": category,
                    "competitors": []
                })
        for comp in competitors:
            if isinstance(comp, dict):
                comp_features = comp.get("features", [])
                for feature in gap_map["data"]["features"]:
                    if feature["name"] in comp_features:
                        feature["competitors"].append(comp["name"])
        for feature in gap_map["data"]["features"]:
            if len(feature["competitors"]) == 0:
                gap_map["data"]["gaps"].append({
                    "feature": feature["name"],
                    "category": feature["category"],
                    "type": "complete_gap"
                })
            elif len(feature["competitors"]) < len(competitors) / 2:
                gap_map["data"]["gaps"].append({
                    "feature": feature["name"],
                    "category": feature["category"],
                    "type": "partial_gap",
                    "competitors_with_feature": feature["competitors"]
                })
    except Exception as e:
        st.warning(f"Error creating feature gap map: {str(e)}")
    return gap_map

class MarketOpportunityMapInput(TypedDict):
    strategy: dict

@tool("create_market_opportunity_map")
def create_market_opportunity_map_tool(input: MarketOpportunityMapInput) -> dict:
    """Creates an interactive market opportunity map from a strategy dictionary."""
    strategy = input["strategy"]
    opportunity_map = {
        "type": "market_opportunity_map",
        "data": {
            "opportunities": [],
            "categories": []
        }
    }
    try:
        if not isinstance(strategy, dict):
            return opportunity_map
        def process_opportunity(opp, source_type):
            if not isinstance(opp, dict):
                return None
            opp_type = opp.get('type', source_type)
            description = opp.get('description') or opp.get('opportunity')
            if not description:
                return None
            category = opp.get('category', 'Uncategorized')
            return {
                "type": opp_type,
                "description": description,
                "category": category
            }
        whitespace_opps = strategy.get("whitespace_opportunities", [])
        for opp in whitespace_opps:
            processed = process_opportunity(opp, 'whitespace')
            if processed:
                opportunity_map["data"]["opportunities"].append(processed)
        innovation_areas = strategy.get("innovation_areas", [])
        for opp in innovation_areas:
            processed = process_opportunity(opp, 'innovation')
            if processed:
                opportunity_map["data"]["opportunities"].append(processed)
        niche_opps = strategy.get("niche_opportunities", [])
        for opp in niche_opps:
            processed = process_opportunity(opp, 'niche')
            if processed:
                opportunity_map["data"]["opportunities"].append(processed)
        if opportunity_map["data"]["opportunities"]:
            opportunity_map["data"]["categories"] = list(set(
                opp["category"] for opp in opportunity_map["data"]["opportunities"]
            ))
    except Exception as e:
        st.error(f"Error creating market opportunity map: {str(e)}")
    return opportunity_map

class CompetitiveLandscapeInput(TypedDict):
    competitors: list

@tool("create_competitive_landscape")
def create_competitive_landscape_tool(input: CompetitiveLandscapeInput) -> dict:
    """Creates an interactive competitive landscape visualization from competitors."""
    competitors = input["competitors"]
    def _normalize_market_share(market_share: str) -> float:
        try:
            match = re.search(r'(\d+(?:\.\d+)?)', market_share)
            if match:
                return float(match.group(1)) / 100.0
            return 0.0
        except:
            return 0.0
    def _normalize_pricing(pricing: str) -> float:
        try:
            match = re.search(r'\$?(\d+(?:\.\d+)?)', pricing)
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return 0.0
    def _categorize_audience(audience: str) -> str:
        audience = audience.lower()
        if any(word in audience for word in ["enterprise", "large", "corporation"]):
            return "Enterprise"
        elif any(word in audience for word in ["small", "medium", "sme", "startup"]):
            return "SMB"
        elif any(word in audience for word in ["individual", "personal", "consumer"]):
            return "Consumer"
        else:
            return "Other"
    landscape = {
        "type": "competitive_landscape",
        "data": {
            "competitors": [],
            "dimensions": ["market_share", "feature_richness", "pricing", "target_audience"]
        }
    }
    try:
        for comp in competitors:
            if isinstance(comp, dict):
                competitor_data = {
                    "name": comp["name"],
                    "position": {
                        "market_share": _normalize_market_share(comp.get("market_share", "N/A")),
                        "feature_richness": len(comp.get("features", [])),
                        "pricing": _normalize_pricing(comp.get("pricing_model", "N/A")),
                        "target_audience": _categorize_audience(comp.get("target_audience", "N/A"))
                    },
                    "metadata": {
                        "website": comp.get("website", "N/A"),
                        "description": comp.get("description", "N/A"),
                        "usp": comp.get("usp", "N/A")
                    }
                }
                landscape["data"]["competitors"].append(competitor_data)
    except Exception as e:
        st.warning(f"Error creating competitive landscape: {str(e)}")
    return landscape

class InnovationRadarInput(TypedDict):
    strategy: dict

@tool("create_innovation_radar")
def create_innovation_radar_tool(input: InnovationRadarInput) -> dict:
    """Creates an innovation radar visualization from a strategy dictionary."""
    strategy = input["strategy"]
    def _assess_innovation_impact(innovation: str) -> str:
        innovation = innovation.lower()
        if any(word in innovation for word in ["revolutionary", "breakthrough", "transformative"]):
            return "High"
        elif any(word in innovation for word in ["significant", "major", "substantial"]):
            return "Medium"
        else:
            return "Low"
    radar = {
        "type": "innovation_radar",
        "data": {
            "categories": [],
            "innovations": []
        }
    }
    try:
        for opp in strategy.get("innovation_areas", []):
            radar["data"]["innovations"].append({
                "category": opp["category"],
                "description": opp["opportunity"],
                "impact": _assess_innovation_impact(opp["opportunity"])
            })
        radar["data"]["categories"] = list(set(
            innovation["category"] for innovation in radar["data"]["innovations"]
        ))
    except Exception as e:
        st.warning(f"Error creating innovation radar: {str(e)}")
    return radar

class VisualGapMapperAgent:
    """Agent responsible for generating interactive visualizations of market gaps and opportunities using LangChain agentic pattern."""
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.tools = [
            create_feature_gap_map_tool,
            create_market_opportunity_map_tool,
            create_competitive_landscape_tool,
            create_innovation_radar_tool
        ]
        self.agent = initialize_agent(
            self.tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True
        )

    def generate_visualization(self, user_query: str, competitors: List[Dict], feature_matrix: Dict, strategy: Dict) -> dict:
        """Use the agent to generate a visualization based on user query and data."""
        input_data = {
            "competitors": competitors,
            "feature_matrix": feature_matrix,
            "strategy": strategy
        }
        prompt = f"""You are a visualization agent. Based on the following user query, select and run the most appropriate visualization tool.\n\nUser Query: {user_query}\n\nAvailable tools: create_feature_gap_map, create_market_opportunity_map, create_competitive_landscape, create_innovation_radar.\n\nData: {input_data}\n\nReturn the visualization data as JSON."""
        try:
            result = self.agent.run(prompt)
            # If result is a string, try to parse as JSON
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    pass
            # If result is not a dict, wrap it in a dict
            if not isinstance(result, dict):
                return {"error": "Visualization agent did not return a dictionary.", "raw_result": str(result)}
            return result
        except Exception as e:
            st.error(f"Agent error: {str(e)}")
            return {"error": str(e)} 