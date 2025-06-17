import streamlit as st
from typing import List, Dict
import json
from langchain_google_genai import ChatGoogleGenerativeAI

class FeatureMatrixBuilderAgent:
    """Agent responsible for building and analyzing feature matrices."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.feature_categories = [
            "Core Features",
            "User Experience",
            "Technical Capabilities",
            "Integration Features",
            "Security & Privacy",
            "Analytics & Reporting",
            "Mobile & Remote Access"
        ]
    
    def build_feature_matrix(self, competitors: List[Dict]) -> Dict:
        """Build a comprehensive feature matrix from competitor data."""
        matrix = {
            "features": {},
            "pricing_comparison": {},
            "audience_segments": {},
            "usps": {}
        }
        
        try:
            # Extract and categorize features
            all_features = set()
            for comp in competitors:
                if isinstance(comp, dict):
                    features = comp.get("features", [])
                    all_features.update(features)
                    
                    # Add to pricing comparison
                    matrix["pricing_comparison"][comp["name"]] = comp.get("pricing_model", "N/A")
                    
                    # Add to audience segments
                    matrix["audience_segments"][comp["name"]] = comp.get("target_audience", "N/A")
                    
                    # Add to USPs
                    matrix["usps"][comp["name"]] = comp.get("usp", "N/A")
            
            # Categorize features
            categorized_features = self._categorize_features(list(all_features))
            matrix["features"] = categorized_features
            
            # Generate feature comparison
            matrix["feature_comparison"] = self._generate_feature_comparison(competitors, categorized_features)
            
            return matrix
        except Exception as e:
            st.error(f"Error building feature matrix: {str(e)}")
            return matrix
    
    def _categorize_features(self, features: List[str]) -> Dict[str, List[str]]:
        """Categorize features into predefined categories."""
        categorized = {category: [] for category in self.feature_categories}
        
        prompt = f"""Categorize the following features into these categories: {', '.join(self.feature_categories)}

Features to categorize:
{json.dumps(features, indent=2)}

Return the response in JSON format with categories as keys and lists of features as values."""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                categories = json.loads(response_text[start_idx:end_idx])
                for category, features in categories.items():
                    if category in categorized:
                        categorized[category] = features
        except Exception as e:
            st.warning(f"Error categorizing features: {str(e)}")
        
        return categorized
    
    def _generate_feature_comparison(self, competitors: List[Dict], categorized_features: Dict[str, List[str]]) -> Dict:
        """Generate a detailed feature comparison matrix."""
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