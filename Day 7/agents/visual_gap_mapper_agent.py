import streamlit as st
from typing import List, Dict
import re
import pandas as pd
import plotly.express as px

class VisualGapMapperAgent:
    """Agent responsible for generating interactive visualizations of market gaps and opportunities."""
    
    def __init__(self):
        self.viz_types = {
            "feature_gaps": "Feature Gap Analysis",
            "market_opportunities": "Market Opportunity Map",
            "competitive_landscape": "Competitive Landscape",
            "innovation_areas": "Innovation Areas"
        }
    
    def generate_visualizations(self, competitors: List[Dict], feature_matrix: Dict, strategy: Dict) -> Dict:
        """Generate comprehensive visualizations of market gaps and opportunities."""
        visualizations = {
            "feature_gap_map": self._create_feature_gap_map(competitors, feature_matrix),
            "market_opportunity_map": self._create_market_opportunity_map(strategy),
            "competitive_landscape": self._create_competitive_landscape(competitors),
            "innovation_radar": self._create_innovation_radar(strategy)
        }
        
        return visualizations
    
    def _create_feature_gap_map(self, competitors: List[Dict], feature_matrix: Dict) -> Dict:
        """Create an interactive feature gap map."""
        gap_map = {
            "type": "feature_gap_map",
            "data": {
                "categories": [],
                "features": [],
                "gaps": []
            }
        }
        
        try:
            # Extract categories and features
            for category, features in feature_matrix["features"].items():
                gap_map["data"]["categories"].append(category)
                for feature in features:
                    gap_map["data"]["features"].append({
                        "name": feature,
                        "category": category,
                        "competitors": []
                    })
            
            # Map features to competitors
            for comp in competitors:
                if isinstance(comp, dict):
                    comp_features = comp.get("features", [])
                    for feature in gap_map["data"]["features"]:
                        if feature["name"] in comp_features:
                            feature["competitors"].append(comp["name"])
            
            # Identify gaps
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
    
    def _create_market_opportunity_map(self, strategy: Dict) -> Dict:
        """Create an interactive market opportunity map."""
        opportunity_map = {
            "type": "market_opportunity_map",
            "data": {
                "opportunities": [],
                "categories": []
            }
        }
        
        try:
            # Process whitespace opportunities
            for opp in strategy.get("whitespace_opportunities", []):
                opportunity_map["data"]["opportunities"].append({
                    "type": opp["type"],
                    "description": opp["description"],
                    "category": "whitespace"
                })
            
            # Process innovation areas
            for opp in strategy.get("innovation_areas", []):
                opportunity_map["data"]["opportunities"].append({
                    "type": "innovation",
                    "description": opp["opportunity"],
                    "category": opp["category"]
                })
            
            # Process niche opportunities
            for opp in strategy.get("niche_opportunities", []):
                opportunity_map["data"]["opportunities"].append({
                    "type": "niche",
                    "description": opp["opportunity"],
                    "category": opp["category"]
                })
            
            # Extract unique categories
            opportunity_map["data"]["categories"] = list(set(
                opp["category"] for opp in opportunity_map["data"]["opportunities"]
            ))
        except Exception as e:
            st.warning(f"Error creating market opportunity map: {str(e)}")
        
        return opportunity_map
    
    def _create_competitive_landscape(self, competitors: List[Dict]) -> Dict:
        """Create an interactive competitive landscape visualization."""
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
                            "market_share": self._normalize_market_share(comp.get("market_share", "N/A")),
                            "feature_richness": len(comp.get("features", [])),
                            "pricing": self._normalize_pricing(comp.get("pricing_model", "N/A")),
                            "target_audience": self._categorize_audience(comp.get("target_audience", "N/A"))
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
    
    def _create_innovation_radar(self, strategy: Dict) -> Dict:
        """Create an innovation radar visualization."""
        radar = {
            "type": "innovation_radar",
            "data": {
                "categories": [],
                "innovations": []
            }
        }
        
        try:
            # Process innovation areas
            for opp in strategy.get("innovation_areas", []):
                radar["data"]["innovations"].append({
                    "category": opp["category"],
                    "description": opp["opportunity"],
                    "impact": self._assess_innovation_impact(opp["opportunity"])
                })
            
            # Extract unique categories
            radar["data"]["categories"] = list(set(
                innovation["category"] for innovation in radar["data"]["innovations"]
            ))
        except Exception as e:
            st.warning(f"Error creating innovation radar: {str(e)}")
        
        return radar
    
    def _normalize_market_share(self, market_share: str) -> float:
        """Normalize market share string to a float value."""
        try:
            # Extract percentage or number from string
            match = re.search(r'(\d+(?:\.\d+)?)', market_share)
            if match:
                return float(match.group(1)) / 100.0
            return 0.0
        except:
            return 0.0
    
    def _normalize_pricing(self, pricing: str) -> float:
        """Normalize pricing string to a float value."""
        try:
            # Extract price from string
            match = re.search(r'\$?(\d+(?:\.\d+)?)', pricing)
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return 0.0
    
    def _categorize_audience(self, audience: str) -> str:
        """Categorize target audience into predefined segments."""
        audience = audience.lower()
        if any(word in audience for word in ["enterprise", "large", "corporation"]):
            return "Enterprise"
        elif any(word in audience for word in ["small", "medium", "sme", "startup"]):
            return "SMB"
        elif any(word in audience for word in ["individual", "personal", "consumer"]):
            return "Consumer"
        else:
            return "Other"
    
    def _assess_innovation_impact(self, innovation: str) -> str:
        """Assess the potential impact of an innovation."""
        # Simple heuristic based on keywords
        innovation = innovation.lower()
        if any(word in innovation for word in ["revolutionary", "breakthrough", "transformative"]):
            return "High"
        elif any(word in innovation for word in ["significant", "major", "substantial"]):
            return "Medium"
        else:
            return "Low" 