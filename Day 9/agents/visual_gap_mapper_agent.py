import streamlit as st
from typing import List, Dict
import re
import pandas as pd
import plotly.express as px


class VisualGapMapperAgent:
    """
    Agent responsible for generating interactive visualizations of market gaps and opportunities.
    
    This agent creates various visualizations including:
    - Feature gap analysis
    - Market opportunity maps
    - Competitive landscape
    - Innovation areas
    """
    
    def __init__(self):
        """Initialize the visualization types available in the agent."""
        self.viz_types = {
            "feature_gaps": "Feature Gap Analysis",
            "market_opportunities": "Market Opportunity Map",
            "competitive_landscape": "Competitive Landscape",
            "innovation_areas": "Innovation Areas"
        }
    
    def generate_visualizations(self, competitors: List[Dict], feature_matrix: Dict, strategy: Dict) -> Dict:
        """
        Generate comprehensive visualizations of market gaps and opportunities.
        
        Args:
            competitors (List[Dict]): List of competitor information
            feature_matrix (Dict): Matrix of features and their categories
            strategy (Dict): Strategic information and opportunities
            
        Returns:
            Dict: Dictionary containing all generated visualizations
        """
        visualizations = {
            "feature_gap_map": self._create_feature_gap_map(competitors, feature_matrix),
            "market_opportunity_map": self._create_market_opportunity_map(strategy),
            "competitive_landscape": self._create_competitive_landscape(competitors),
            "innovation_radar": self._create_innovation_radar(strategy)
        }
        return visualizations
    
    def _create_feature_gap_map(self, competitors: List[Dict], feature_matrix: Dict) -> Dict:
        """
        Create an interactive feature gap map.
        
        Args:
            competitors (List[Dict]): List of competitor information
            feature_matrix (Dict): Matrix of features and their categories
            
        Returns:
            Dict: Feature gap map visualization data
        """
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
        """
        Create an interactive market opportunity map.
        
        Args:
            strategy (Dict): Strategic information and opportunities
            
        Returns:
            Dict: Market opportunity map visualization data
        """
        opportunity_map = {
            "type": "market_opportunity_map",
            "data": {
                "opportunities": [],
                "categories": []
            }
        }
        
        try:
            # Validate strategy input
            if not isinstance(strategy, dict):
                st.error("Strategy must be a dictionary")
                return opportunity_map
                
            # Debug information about available data
            available_keys = list(strategy.keys())
            st.info(f"Available strategy keys: {available_keys}")
            
            def process_opportunity(opp, source_type):
                """Helper function to process and validate opportunities"""
                if not isinstance(opp, dict):
                    return None
                    
                # Extract or determine type
                opp_type = opp.get('type', source_type)
                
                # Get description from either description or opportunity field
                description = opp.get('description') or opp.get('opportunity')
                if not description:
                    return None
                    
                # Get or determine category
                category = opp.get('category', 'Uncategorized')
                
                return {
                    "type": opp_type,
                    "description": description,
                    "category": category
                }
            
            # Process whitespace opportunities
            whitespace_opps = strategy.get("whitespace_opportunities", [])
            if whitespace_opps:
                st.info(f"Found {len(whitespace_opps)} whitespace opportunities")
                for opp in whitespace_opps:
                    processed = process_opportunity(opp, 'whitespace')
                    if processed:
                        opportunity_map["data"]["opportunities"].append(processed)
                    else:
                        st.warning(f"Skipping invalid whitespace opportunity: {opp}")
            
            # Process innovation areas
            innovation_areas = strategy.get("innovation_areas", [])
            if innovation_areas:
                st.info(f"Found {len(innovation_areas)} innovation areas")
                for opp in innovation_areas:
                    processed = process_opportunity(opp, 'innovation')
                    if processed:
                        opportunity_map["data"]["opportunities"].append(processed)
                    else:
                        st.warning(f"Skipping invalid innovation area: {opp}")
            
            # Process niche opportunities
            niche_opps = strategy.get("niche_opportunities", [])
            if niche_opps:
                st.info(f"Found {len(niche_opps)} niche opportunities")
                for opp in niche_opps:
                    processed = process_opportunity(opp, 'niche')
                    if processed:
                        opportunity_map["data"]["opportunities"].append(processed)
                    else:
                        st.warning(f"Skipping invalid niche opportunity: {opp}")
            
            # Extract unique categories only if we have opportunities
            if opportunity_map["data"]["opportunities"]:
                opportunity_map["data"]["categories"] = list(set(
                    opp["category"] for opp in opportunity_map["data"]["opportunities"]
                ))
                st.success(f"Successfully processed {len(opportunity_map['data']['opportunities'])} opportunities across {len(opportunity_map['data']['categories'])} categories")
            else:
                st.warning("No valid opportunities found in the strategy data. Please ensure the strategy contains valid opportunities with at least a description/opportunity field.")
                
        except Exception as e:
            st.error(f"Error creating market opportunity map: {str(e)}")
            # Ensure we return a valid structure even in case of error
            opportunity_map["data"]["opportunities"] = []
            opportunity_map["data"]["categories"] = []
        
        return opportunity_map
    
    def _create_competitive_landscape(self, competitors: List[Dict]) -> Dict:
        """
        Create an interactive competitive landscape visualization.
        
        Args:
            competitors (List[Dict]): List of competitor information
            
        Returns:
            Dict: Competitive landscape visualization data
        """
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
        """
        Create an innovation radar visualization.
        
        Args:
            strategy (Dict): Strategic information and opportunities
            
        Returns:
            Dict: Innovation radar visualization data
        """
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
        """
        Normalize market share string to a float value.
        
        Args:
            market_share (str): Market share as a string
            
        Returns:
            float: Normalized market share value
        """
        try:
            match = re.search(r'(\d+(?:\.\d+)?)', market_share)
            if match:
                return float(match.group(1)) / 100.0
            return 0.0
        except:
            return 0.0
    
    def _normalize_pricing(self, pricing: str) -> float:
        """
        Normalize pricing string to a float value.
        
        Args:
            pricing (str): Pricing information as a string
            
        Returns:
            float: Normalized pricing value
        """
        try:
            match = re.search(r'\$?(\d+(?:\.\d+)?)', pricing)
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return 0.0
    
    def _categorize_audience(self, audience: str) -> str:
        """
        Categorize target audience into predefined segments.
        
        Args:
            audience (str): Target audience description
            
        Returns:
            str: Categorized audience segment
        """
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
        """
        Assess the potential impact of an innovation.
        
        Args:
            innovation (str): Innovation description
            
        Returns:
            str: Impact level (High/Medium/Low)
        """
        innovation = innovation.lower()
        if any(word in innovation for word in ["revolutionary", "breakthrough", "transformative"]):
            return "High"
        elif any(word in innovation for word in ["significant", "major", "substantial"]):
            return "Medium"
        else:
            return "Low" 