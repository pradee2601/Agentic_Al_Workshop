import streamlit as st
from typing import List, Dict, Optional
import json
import re
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from typing_extensions import TypedDict

# --- Tool Definitions ---

PLATFORMS = {
    "product_hunt": "https://www.producthunt.com/search?q=",
    "crunchbase": "https://www.crunchbase.com/search/companies?q=",
    "g2": "https://www.g2.com/search?q="
}

class SearchPlatformInput(TypedDict):
    startup_idea: str
    platform: str

@tool("search_platform")
def search_platform_tool(input: SearchPlatformInput, search: TavilySearch = None) -> List[Dict]:
    """Searches a specific platform for competitors based on the startup idea and platform name."""
    startup_idea = input["startup_idea"]
    platform = input["platform"]
    if not startup_idea or not platform:
        return []
    base_url = PLATFORMS.get(platform)
    if not base_url:
        return []
    search_query = f"{startup_idea} site:{base_url}"
    try:
        results = search.invoke({"query": search_query})
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        return results
    except Exception as e:
        st.warning(f"Error searching {platform}: {str(e)}")
        return []

class ExtractCompetitorInfoInput(TypedDict):
    result: dict
    platform: str

@tool("extract_competitor_info")
def extract_competitor_info_tool(input: ExtractCompetitorInfoInput) -> Optional[Dict]:
    """Extracts competitor information from a search result and platform name."""
    result = input["result"]
    platform = input.get("platform", "general")
    try:
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")
        if any(site in url.lower() for site in ["capterra", "softwareworld", "alternatives", "comparison", "review"]):
            return None
        names = _extract_competitor_names(title, content)
        if not names:
            return None
        name = names[0]
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

class EnrichWithRagInput(TypedDict):
    competitor: dict
    startup_idea: str

@tool("enrich_with_rag")
def enrich_with_rag_tool(input: EnrichWithRagInput, vector_db: Chroma = None, llm: ChatGoogleGenerativeAI = None) -> Dict:
    """Enriches a competitor dictionary with additional information using RAG (vector DB and LLM)."""
    competitor = input["competitor"]
    startup_idea = input.get("startup_idea", "")
    if not competitor:
        return competitor
    try:
        similar_companies = vector_db.similarity_search(competitor["name"], k=3) if vector_db else []
        enriched_comp = competitor.copy()
        if similar_companies:
            for similar in similar_companies:
                if isinstance(similar, dict):
                    for key, value in similar.items():
                        if key not in enriched_comp or enriched_comp[key] == "N/A":
                            enriched_comp[key] = value
        if llm:
            missing_fields = [field for field, value in enriched_comp.items() if value in ["N/A", [], {}]]
            if missing_fields:
                prompt = f"""Based on the following information about {enriched_comp['name']}, generate realistic and plausible information for the missing fields: {', '.join(missing_fields)}.\n\nCompany Information:\n{json.dumps(enriched_comp, indent=2)}\n\nStartup Idea: {startup_idea}\n\nPlease generate information that would be realistic for this type of company. Return the response in JSON format with only the missing fields."""
                try:
                    response = llm.invoke(prompt)
                    response_text = response.content
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        gen_json = json.loads(response_text[start_idx:end_idx])
                        for field in missing_fields:
                            if field in gen_json:
                                enriched_comp[field] = gen_json[field]
                except Exception as e:
                    st.warning(f"Could not generate missing information for {enriched_comp['name']}: {str(e)}")
        return enriched_comp
    except Exception as e:
        st.warning(f"Error enriching competitor {competitor.get('name', 'Unknown')}: {str(e)}")
        return competitor

def _extract_competitor_names(title: str, content: str) -> List[str]:
    names = []
    patterns = [
        r"best\s+([A-Za-z0-9\s]+?)(?:\s+app|\s+software|\s+system|\s+tool|\s+device)",
        r"top\s+([A-Za-z0-9\s]+?)(?:\s+app|\s+software|\s+system|\s+tool|\s+device)",
        r"([A-Za-z0-9\s]+?)(?:\s+vs\.|\s+versus|\s+compared to|\s+alternative to)",
        r"([A-Za-z0-9\s]+?)(?:\s+review|\s+guide|\s+comparison)",
        r"competitors?\s+include\s+([A-Za-z0-9\s,]+?)(?:\.|\n|$)",
        r"alternatives?\s+include\s+([A-Za-z0-9\s,]+?)(?:\.|\n|$)"
    ]
    for pattern in patterns:
        matches = re.finditer(pattern, title, re.IGNORECASE)
        for match in matches:
            name = match.group(1).strip()
            if name and len(name) > 2:
                names.append(name)
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            name = match.group(1).strip()
            if name and len(name) > 2:
                names.append(name)
    cleaned_names = []
    for name in names:
        name = re.sub(r'^(the|a|an)\s+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+(app|software|system|tool|device)$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[^\w\s-]', '', name)
        name = ' '.join(name.split())
        if name and len(name) > 2:
            cleaned_names.append(name)
    return list(set(cleaned_names))

# --- Agent Runner ---
def run_competitor_discovery_agent(startup_idea: str, search: TavilySearch, llm: ChatGoogleGenerativeAI, vector_db: Chroma) -> List[Dict]:
    # Bind search and vector_db/llm to tools using partials
    from functools import partial
    tools = [
        search_platform_tool.bind(search=search),
        extract_competitor_info_tool,
        enrich_with_rag_tool.bind(vector_db=vector_db, llm=llm)
    ]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True
    )
    agent_prompt = f"""
    Your task is to discover competitors for the following startup idea: '{startup_idea}'.\n
    1. For each platform in ['product_hunt', 'crunchbase', 'g2'], use the search_platform tool to get a list of search results.\n2. For each search result (a dict), use extract_competitor_info, passing the result as the 'result' argument and the platform as 'platform'.\n3. For each valid competitor dict returned, use enrich_with_rag, passing the competitor dict as 'competitor' and the startup idea as 'startup_idea'.\n4. If no competitors are found, do a general web search for 'top {startup_idea} competitors alternatives' and repeat.\n5. Return a list of enriched competitor dicts as JSON.
    """
    try:
        response = agent.run(agent_prompt)
        if isinstance(response, str):
            try:
                competitors = json.loads(response)
                if isinstance(competitors, list):
                    competitors = [c for c in competitors if isinstance(c, dict)]
                    return competitors
            except Exception:
                pass
        if isinstance(response, list):
            response = [c for c in response if isinstance(c, dict)]
        return response
    except Exception as e:
        st.error(f"Agent error: {str(e)}")
        return [] 