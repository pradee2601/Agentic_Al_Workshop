import streamlit as st
from typing import List, Dict, Optional
import json
import re
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

class CompetitorDiscoveryAgent:
    """Agent responsible for discovering and analyzing competitors using RAG."""
    
    def __init__(self, search: TavilySearch, llm: ChatGoogleGenerativeAI, vector_db: Chroma):
        self.search = search
        self.llm = llm
        self.vector_db = vector_db
        self.platforms = {
            "product_hunt": "https://www.producthunt.com/search?q=",
            "crunchbase": "https://www.crunchbase.com/search/companies?q=",
            "g2": "https://www.g2.com/search?q="
        }
    
    def discover_competitors(self, startup_idea: str) -> List[Dict]:
        """Discover competitors from multiple platforms using RAG."""
        competitors = []
        
        # Search across platforms
        for platform, base_url in self.platforms.items():
            search_query = f"{startup_idea} site:{base_url}"
            try:
                results = self.search.invoke({"query": search_query})
                
                if isinstance(results, dict) and "results" in results:
                    platform_results = results["results"]
                else:
                    platform_results = results
                    
                # Process results
                for result in platform_results:
                    if isinstance(result, dict):
                        competitor = self._extract_competitor_info(result, platform)
                        if competitor:
                            competitors.append(competitor)
            except Exception as e:
                st.warning(f"Error searching {platform}: {str(e)}")
                continue
        
        # If no competitors found, try a broader search
        if not competitors:
            try:
                results = self.search.invoke({"query": f"top {startup_idea} competitors alternatives"})
                if isinstance(results, dict) and "results" in results:
                    platform_results = results["results"]
                else:
                    platform_results = results
                    
                for result in platform_results:
                    if isinstance(result, dict):
                        competitor = self._extract_competitor_info(result, "general")
                        if competitor:
                            competitors.append(competitor)
            except Exception as e:
                st.warning(f"Error in broader search: {str(e)}")
        
        # Enrich with RAG
        enriched_competitors = self._enrich_with_rag(competitors, startup_idea)
        return enriched_competitors
    
    def _extract_competitor_info(self, result: Dict, platform: str) -> Optional[Dict]:
        """Extract competitor information from search result."""
        try:
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            
            # Skip if it's a comparison or review site
            if any(site in url.lower() for site in ["capterra", "softwareworld", "alternatives", "comparison", "review"]):
                return None
            
            # Extract name using existing function
            names = self._extract_competitor_names(title, content)
            if not names:
                return None
                
            name = names[0]  # Use first matched name
            
            # Clean up the description
            description = content[:200] + "..." if content else "No description available"
            description = re.sub(r'\s+', ' ', description).strip()
            
            return {
                "name": name,
                "website": url if url else "N/A",
                "description": description,
                "platform": platform,
                "features": [],
                "pricing_model": "N/A",
                "target_audience": "N/A",
                "usp": "N/A",
                "market_share": "N/A",
                "funding_status": "N/A",
                "user_rating": "N/A",
                "feature_categories": {}
            }
        except Exception as e:
            st.warning(f"Error extracting competitor info: {str(e)}")
            return None
    
    def _extract_competitor_names(self, title: str, content: str) -> List[str]:
        """Extract competitor names from title and content."""
        names = []
        
        # Common patterns for finding competitor names
        patterns = [
            r"best\s+([A-Za-z0-9\s]+?)(?:\s+app|\s+software|\s+system|\s+tool|\s+device)",
            r"top\s+([A-Za-z0-9\s]+?)(?:\s+app|\s+software|\s+system|\s+tool|\s+device)",
            r"([A-Za-z0-9\s]+?)(?:\s+vs\.|\s+versus|\s+compared to|\s+alternative to)",
            r"([A-Za-z0-9\s]+?)(?:\s+review|\s+guide|\s+comparison)",
            r"competitors?\s+include\s+([A-Za-z0-9\s,]+?)(?:\.|\n|$)",
            r"alternatives?\s+include\s+([A-Za-z0-9\s,]+?)(?:\.|\n|$)"
        ]
        
        # Extract from title
        for pattern in patterns:
            matches = re.finditer(pattern, title, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if name and len(name) > 2:
                    names.append(name)
        
        # Extract from content
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if name and len(name) > 2:
                    names.append(name)
        
        # Clean up names
        cleaned_names = []
        for name in names:
            # Remove common prefixes/suffixes
            name = re.sub(r'^(the|a|an)\s+', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s+(app|software|system|tool|device)$', '', name, flags=re.IGNORECASE)
            # Remove special characters
            name = re.sub(r'[^\w\s-]', '', name)
            # Remove extra whitespace
            name = ' '.join(name.split())
            if name and len(name) > 2:
                cleaned_names.append(name)
        
        return list(set(cleaned_names))  # Remove duplicates
    
    def _enrich_with_rag(self, competitors: List[Dict], startup_idea: str) -> List[Dict]:
        """Enrich competitor information using RAG."""
        enriched_competitors = []
        
        for comp in competitors:
            try:
                # Query vector database for similar companies
                similar_companies = self.vector_db.similarity_search(comp["name"], k=3)
                
                # Combine information
                enriched_comp = comp.copy()
                if similar_companies:
                    # Merge information from similar companies
                    for similar in similar_companies:
                        if isinstance(similar, dict):
                            for key, value in similar.items():
                                if key not in enriched_comp or enriched_comp[key] == "N/A":
                                    enriched_comp[key] = value
                
                # Use LLM to fill missing information
                if self.llm:
                    enriched_comp = self._fill_missing_info(enriched_comp, startup_idea)
                
                enriched_competitors.append(enriched_comp)
            except Exception as e:
                st.warning(f"Error enriching competitor {comp.get('name', 'Unknown')}: {str(e)}")
                enriched_competitors.append(comp)
        
        return enriched_competitors
    
    def _fill_missing_info(self, competitor: Dict, startup_idea: str) -> Dict:
        """Fill missing information using LLM."""
        missing_fields = []
        for field, value in competitor.items():
            if value in ["N/A", [], {}]:
                missing_fields.append(field)
        
        if missing_fields:
            prompt = f"""Based on the following information about {competitor['name']}, generate realistic and plausible information for the missing fields: {', '.join(missing_fields)}.

Company Information:
{json.dumps(competitor, indent=2)}

Startup Idea: {startup_idea}

Please generate information that would be realistic for this type of company. Return the response in JSON format with only the missing fields."""
            
            try:
                response = self.llm.invoke(prompt)
                response_text = response.content
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    gen_json = json.loads(response_text[start_idx:end_idx])
                    for field in missing_fields:
                        if field in gen_json:
                            competitor[field] = gen_json[field]
            except Exception as e:
                st.warning(f"Could not generate missing information for {competitor['name']}: {str(e)}")
        
        return competitor 