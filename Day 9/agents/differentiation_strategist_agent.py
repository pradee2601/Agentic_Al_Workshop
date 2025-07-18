import streamlit as st
from typing import List, Dict
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from typing_extensions import TypedDict

class IdentifyWhitespaceInput(TypedDict):
    competitors: list
    feature_matrix: dict

@tool("identify_whitespace")
def identify_whitespace_tool(input: IdentifyWhitespaceInput, llm=None) -> list:
    """Identifies whitespace opportunities in the market using an LLM."""
    competitors = input["competitors"]
    feature_matrix = input["feature_matrix"]
    whitespace = []
    try:
        prompt = f"""Based on the following competitors and their features, identify potential whitespace opportunities in the market.\n\nCompetitors and Features:\n{json.dumps(feature_matrix, indent=2)}\n\nReturn the response in JSON format with the following structure:\n{{\n    'feature_gaps': ['List of potential feature gaps'],\n    'market_gaps': ['List of potential market gaps'],\n    'opportunity_areas': ['List of opportunity areas']\n}}"""
        response = llm.invoke(prompt)
        response_text = response.content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            gaps = json.loads(response_text[start_idx:end_idx])
            whitespace.extend([
                {"type": "feature_gap", "description": gap}
                for gap in gaps.get("feature_gaps", [])
            ])
            whitespace.extend([
                {"type": "market_gap", "description": gap}
                for gap in gaps.get("market_gaps", [])
            ])
            whitespace.extend([
                {"type": "opportunity", "description": gap}
                for gap in gaps.get("opportunity_areas", [])
            ])
    except Exception as e:
        # Optionally log or handle error
        pass
    return whitespace

class IdentifyInnovationAreasInput(TypedDict):
    startup_idea: str
    competitors: list

@tool("identify_innovation_areas")
def identify_innovation_areas_tool(input: IdentifyInnovationAreasInput, llm=None) -> list:
    """Identifies potential areas for innovation using an LLM."""
    startup_idea = input["startup_idea"]
    competitors = input["competitors"]
    innovation_areas = []
    try:
        prompt = f"""Based on the startup idea '{startup_idea}' and these competitors:\n{json.dumps(competitors, indent=2)}\n\nIdentify potential areas for innovation in the following categories:\n1. Technical Innovation\n2. Feature Innovation\n3. Business Model Innovation\n4. User Experience Innovation\n\nReturn the response in JSON format with categories as keys and lists of innovation opportunities as values."""
        response = llm.invoke(prompt)
        response_text = response.content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            innovations = json.loads(response_text[start_idx:end_idx])
            for category, opportunities in innovations.items():
                innovation_areas.extend([
                    {"category": category, "opportunity": opp}
                    for opp in opportunities
                ])
    except Exception as e:
        # Optionally log or handle error
        pass
    return innovation_areas

class AnalyzePricingOpportunitiesInput(TypedDict):
    competitors: list

@tool("analyze_pricing_opportunities")
def analyze_pricing_opportunities_tool(input: AnalyzePricingOpportunitiesInput, llm=None) -> list:
    """Analyzes pricing opportunities and potential disruptions using an LLM."""
    competitors = input["competitors"]
    pricing_opportunities = []
    try:
        pricing_models = {}
        for comp in competitors:
            if isinstance(comp, dict):
                pricing_models[comp["name"]] = comp.get("pricing_model", "N/A")
        prompt = f"""Based on these competitor pricing models:\n{json.dumps(pricing_models, indent=2)}\n\nIdentify potential pricing opportunities and disruptions in the following areas:\n1. Pricing Model Innovation\n2. Value-Based Pricing Opportunities\n3. Market Positioning Opportunities\n4. Revenue Model Innovation\n\nReturn the response in JSON format with categories as keys and lists of opportunities as values."""
        response = llm.invoke(prompt)
        response_text = response.content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            opportunities = json.loads(response_text[start_idx:end_idx])
            for category, opps in opportunities.items():
                pricing_opportunities.extend([
                    {"category": category, "opportunity": opp}
                    for opp in opps
                ])
    except Exception as e:
        # Optionally log or handle error
        pass
    return pricing_opportunities

class IdentifyNicheOpportunitiesInput(TypedDict):
    competitors: list

@tool("identify_niche_opportunities")
def identify_niche_opportunities_tool(input: IdentifyNicheOpportunitiesInput, llm=None) -> list:
    """Identifies potential niche market opportunities using an LLM."""
    competitors = input["competitors"]
    niche_opportunities = []
    try:
        audiences = {}
        for comp in competitors:
            if isinstance(comp, dict):
                audiences[comp["name"]] = comp.get("target_audience", "N/A")
        prompt = f"""Based on these competitor target audiences:\n{json.dumps(audiences, indent=2)}\n\nIdentify potential niche market opportunities in the following areas:\n1. Underserved Segments\n2. Emerging Markets\n3. Specialized Use Cases\n4. Geographic Opportunities\n\nReturn the response in JSON format with categories as keys and lists of opportunities as values."""
        response = llm.invoke(prompt)
        response_text = response.content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            opportunities = json.loads(response_text[start_idx:end_idx])
            for category, opps in opportunities.items():
                niche_opportunities.extend([
                    {"category": category, "opportunity": opp}
                    for opp in opps
                ])
    except Exception as e:
        # Optionally log or handle error
        pass
    return niche_opportunities

class GeneratePositioningStrategyInput(TypedDict):
    startup_idea: str
    competitors: list

@tool("generate_positioning_strategy")
def generate_positioning_strategy_tool(input: GeneratePositioningStrategyInput, llm=None) -> dict:
    """Generates strategic positioning recommendations using an LLM."""
    startup_idea = input["startup_idea"]
    competitors = input["competitors"]
    positioning = {
        "market_position": "",
        "value_proposition": "",
        "key_differentiators": [],
        "target_audience": "",
        "brand_positioning": ""
    }
    try:
        prompt = f"""Based on the startup idea '{startup_idea}' and these competitors:\n{json.dumps(competitors, indent=2)}\n\nGenerate a strategic positioning strategy in the following JSON format:\n{{\n    'market_position': 'Clear market position statement',\n    'value_proposition': 'Compelling value proposition',\n    'key_differentiators': ['List of key differentiators'],\n    'target_audience': 'Specific target audience description',\n    'brand_positioning': 'Brand positioning statement'\n}}"""
        response = llm.invoke(prompt)
        response_text = response.content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            strategy = json.loads(response_text[start_idx:end_idx])
            positioning.update(strategy)
    except Exception as e:
        # Optionally log or handle error
        pass
    return positioning

class IdentifyCompetitiveAdvantagesInput(TypedDict):
    startup_idea: str
    competitors: list

@tool("identify_competitive_advantages")
def identify_competitive_advantages_tool(input: IdentifyCompetitiveAdvantagesInput, llm=None) -> list:
    """Identifies potential competitive advantages using an LLM."""
    startup_idea = input["startup_idea"]
    competitors = input["competitors"]
    advantages = []
    try:
        prompt = f"""Based on the startup idea '{startup_idea}' and these competitors:\n{json.dumps(competitors, indent=2)}\n\nIdentify potential competitive advantages in the following areas:\n1. Technical Advantages\n2. Feature Advantages\n3. Market Advantages\n4. Operational Advantages\n5. Strategic Advantages\n\nReturn the response in JSON format with categories as keys and lists of advantages as values."""
        response = llm.invoke(prompt)
        response_text = response.content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            comp_advantages = json.loads(response_text[start_idx:end_idx])
            for category, advs in comp_advantages.items():
                advantages.extend([
                    {"category": category, "advantage": adv}
                    for adv in advs
                ])
    except Exception as e:
        # Optionally log or handle error
        pass
    return advantages

class DifferentiationStrategistAgent:
    """Agent responsible for identifying differentiation opportunities and strategic positioning."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.strategy_categories = [
            "Feature Innovation",
            "Pricing Strategy",
            "Niche Targeting",
            "Technical Advantage",
            "User Experience",
            "Market Positioning"
        ]
    
    def generate_strategy(self, startup_idea: str, competitors: List[Dict], feature_matrix: Dict) -> Dict:
        """Generate comprehensive differentiation strategy."""
        strategy = {
            "whitespace_opportunities": self._identify_whitespace(competitors, feature_matrix),
            "innovation_areas": self._identify_innovation_areas(startup_idea, competitors),
            "pricing_opportunities": self._analyze_pricing_opportunities(competitors),
            "niche_opportunities": self._identify_niche_opportunities(competitors),
            "positioning_strategy": self._generate_positioning_strategy(startup_idea, competitors),
            "competitive_advantages": self._identify_competitive_advantages(startup_idea, competitors)
        }
        
        return strategy
    
    def _identify_whitespace(self, competitors: List[Dict], feature_matrix: Dict) -> List[Dict]:
        """Identify whitespace opportunities in the market."""
        whitespace = []
        
        try:
            # Analyze feature gaps
            all_features = set()
            for comp in competitors:
                if isinstance(comp, dict):
                    all_features.update(comp.get("features", []))
            
            # Identify potential feature gaps
            prompt = f"""Based on the following competitors and their features, identify potential whitespace opportunities in the market.

Competitors and Features:
{json.dumps(feature_matrix, indent=2)}

Return the response in JSON format with the following structure:
{{
    "feature_gaps": ["List of potential feature gaps"],
    "market_gaps": ["List of potential market gaps"],
    "opportunity_areas": ["List of opportunity areas"]
}}"""
            
            response = self.llm.invoke(prompt)
            response_text = response.content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                gaps = json.loads(response_text[start_idx:end_idx])
                whitespace.extend([
                    {"type": "feature_gap", "description": gap}
                    for gap in gaps.get("feature_gaps", [])
                ])
                whitespace.extend([
                    {"type": "market_gap", "description": gap}
                    for gap in gaps.get("market_gaps", [])
                ])
                whitespace.extend([
                    {"type": "opportunity", "description": gap}
                    for gap in gaps.get("opportunity_areas", [])
                ])
        except Exception as e:
            st.warning(f"Error identifying whitespace: {str(e)}")
        
        return whitespace
    
    def _identify_innovation_areas(self, startup_idea: str, competitors: List[Dict]) -> List[Dict]:
        """Identify potential areas for innovation."""
        innovation_areas = []
        
        try:
            prompt = f"""Based on the startup idea '{startup_idea}' and these competitors:
{json.dumps(competitors, indent=2)}

Identify potential areas for innovation in the following categories:
1. Technical Innovation
2. Feature Innovation
3. Business Model Innovation
4. User Experience Innovation

Return the response in JSON format with categories as keys and lists of innovation opportunities as values."""
            
            response = self.llm.invoke(prompt)
            response_text = response.content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                innovations = json.loads(response_text[start_idx:end_idx])
                for category, opportunities in innovations.items():
                    innovation_areas.extend([
                        {"category": category, "opportunity": opp}
                        for opp in opportunities
                    ])
        except Exception as e:
            st.warning(f"Error identifying innovation areas: {str(e)}")
        
        return innovation_areas
    
    def _analyze_pricing_opportunities(self, competitors: List[Dict]) -> List[Dict]:
        """Analyze pricing opportunities and potential disruptions."""
        pricing_opportunities = []
        
        try:
            # Extract pricing information
            pricing_models = {}
            for comp in competitors:
                if isinstance(comp, dict):
                    pricing_models[comp["name"]] = comp.get("pricing_model", "N/A")
            
            prompt = f"""Based on these competitor pricing models:
{json.dumps(pricing_models, indent=2)}

Identify potential pricing opportunities and disruptions in the following areas:
1. Pricing Model Innovation
2. Value-Based Pricing Opportunities
3. Market Positioning Opportunities
4. Revenue Model Innovation

Return the response in JSON format with categories as keys and lists of opportunities as values."""
            
            response = self.llm.invoke(prompt)
            response_text = response.content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                opportunities = json.loads(response_text[start_idx:end_idx])
                for category, opps in opportunities.items():
                    pricing_opportunities.extend([
                        {"category": category, "opportunity": opp}
                        for opp in opps
                    ])
        except Exception as e:
            st.warning(f"Error analyzing pricing opportunities: {str(e)}")
        
        return pricing_opportunities
    
    def _identify_niche_opportunities(self, competitors: List[Dict]) -> List[Dict]:
        """Identify potential niche market opportunities."""
        niche_opportunities = []
        
        try:
            # Extract target audience information
            audiences = {}
            for comp in competitors:
                if isinstance(comp, dict):
                    audiences[comp["name"]] = comp.get("target_audience", "N/A")
            
            prompt = f"""Based on these competitor target audiences:
{json.dumps(audiences, indent=2)}

Identify potential niche market opportunities in the following areas:
1. Underserved Segments
2. Emerging Markets
3. Specialized Use Cases
4. Geographic Opportunities

Return the response in JSON format with categories as keys and lists of opportunities as values."""
            
            response = self.llm.invoke(prompt)
            response_text = response.content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                opportunities = json.loads(response_text[start_idx:end_idx])
                for category, opps in opportunities.items():
                    niche_opportunities.extend([
                        {"category": category, "opportunity": opp}
                        for opp in opps
                    ])
        except Exception as e:
            st.warning(f"Error identifying niche opportunities: {str(e)}")
        
        return niche_opportunities
    
    def _generate_positioning_strategy(self, startup_idea: str, competitors: List[Dict]) -> Dict:
        """Generate strategic positioning recommendations."""
        positioning = {
            "market_position": "",
            "value_proposition": "",
            "key_differentiators": [],
            "target_audience": "",
            "brand_positioning": ""
        }
        
        try:
            prompt = f"""Based on the startup idea '{startup_idea}' and these competitors:
{json.dumps(competitors, indent=2)}

Generate a strategic positioning strategy in the following JSON format:
{{
    "market_position": "Clear market position statement",
    "value_proposition": "Compelling value proposition",
    "key_differentiators": ["List of key differentiators"],
    "target_audience": "Specific target audience description",
    "brand_positioning": "Brand positioning statement"
}}"""
            
            response = self.llm.invoke(prompt)
            response_text = response.content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                strategy = json.loads(response_text[start_idx:end_idx])
                positioning.update(strategy)
        except Exception as e:
            st.warning(f"Error generating positioning strategy: {str(e)}")
        
        return positioning
    
    def _identify_competitive_advantages(self, startup_idea: str, competitors: List[Dict]) -> List[Dict]:
        """Identify potential competitive advantages."""
        advantages = []
        
        try:
            prompt = f"""Based on the startup idea '{startup_idea}' and these competitors:
{json.dumps(competitors, indent=2)}

Identify potential competitive advantages in the following areas:
1. Technical Advantages
2. Feature Advantages
3. Market Advantages
4. Operational Advantages
5. Strategic Advantages

Return the response in JSON format with categories as keys and lists of advantages as values."""
            
            response = self.llm.invoke(prompt)
            response_text = response.content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                comp_advantages = json.loads(response_text[start_idx:end_idx])
                for category, advs in comp_advantages.items():
                    advantages.extend([
                        {"category": category, "advantage": adv}
                        for adv in advs
                    ])
        except Exception as e:
            st.warning(f"Error identifying competitive advantages: {str(e)}")
        
        return advantages 