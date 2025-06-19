from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from agents.competitor_discovery_agent import CompetitorDiscoveryAgent
from agents.feature_matrix_builder_agent import FeatureMatrixBuilderAgent
from agents.differentiation_strategist_agent import DifferentiationStrategistAgent
from agents.visual_gap_mapper_agent import VisualGapMapperAgent

# Define the state type
class MarketAnalysisState(TypedDict):
    startup_idea: str
    competitors: List[Dict]
    feature_matrix: Dict
    strategy: Dict
    visualizations: Dict
    error: str

def create_market_analysis_graph(
    search: TavilySearch,
    llm: ChatGoogleGenerativeAI,
    vector_db: Chroma
) -> Graph:
    """Create the market analysis workflow graph."""
    
    # Initialize agents
    competitor_discovery = CompetitorDiscoveryAgent(search, llm, vector_db)
    feature_matrix_builder = FeatureMatrixBuilderAgent(llm)
    differentiation_strategist = DifferentiationStrategistAgent(llm)
    visual_gap_mapper = VisualGapMapperAgent()
    
    # Define the nodes
    def discover_competitors(state: MarketAnalysisState) -> MarketAnalysisState:
        try:
            competitors = competitor_discovery.discover_competitors(state["startup_idea"])
            return {**state, "competitors": competitors}
        except Exception as e:
            return {**state, "error": f"Error discovering competitors: {str(e)}"}
    
    def build_feature_matrix(state: MarketAnalysisState) -> MarketAnalysisState:
        try:
            feature_matrix = feature_matrix_builder.build_feature_matrix(state["competitors"])
            return {**state, "feature_matrix": feature_matrix}
        except Exception as e:
            return {**state, "error": f"Error building feature matrix: {str(e)}"}
    
    def generate_strategy(state: MarketAnalysisState) -> MarketAnalysisState:
        try:
            strategy = differentiation_strategist.generate_strategy(
                state["startup_idea"],
                state["competitors"],
                state["feature_matrix"]
            )
            return {**state, "strategy": strategy}
        except Exception as e:
            return {**state, "error": f"Error generating strategy: {str(e)}"}
    
    def create_visualizations(state: MarketAnalysisState) -> MarketAnalysisState:
        try:
            visualizations = visual_gap_mapper.generate_visualizations(
                state["competitors"],
                state["feature_matrix"],
                state["strategy"]
            )
            return {**state, "visualizations": visualizations}
        except Exception as e:
            return {**state, "error": f"Error creating visualizations: {str(e)}"}
    
    # Create the graph
    workflow = StateGraph(MarketAnalysisState)
    
    # Add nodes
    workflow.add_node("discover_competitors", discover_competitors)
    workflow.add_node("build_feature_matrix", build_feature_matrix)
    workflow.add_node("generate_strategy", generate_strategy)
    workflow.add_node("create_visualizations", create_visualizations)
    
    # Define edges
    workflow.add_edge("discover_competitors", "build_feature_matrix")
    workflow.add_edge("build_feature_matrix", "generate_strategy")
    workflow.add_edge("generate_strategy", "create_visualizations")
    
    # Set entry and exit points
    workflow.set_entry_point("discover_competitors")
    workflow.set_finish_point("create_visualizations")
    
    return workflow.compile()

def run_market_analysis(
    startup_idea: str,
    search: TavilySearch,
    llm: ChatGoogleGenerativeAI,
    vector_db: Chroma
) -> Dict:
    """Run the market analysis workflow."""
    
    # Create the graph
    graph = create_market_analysis_graph(search, llm, vector_db)
    
    # Initialize the state
    initial_state = {
        "startup_idea": startup_idea,
        "competitors": [],
        "feature_matrix": {},
        "strategy": {},
        "visualizations": {},
        "error": ""
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    return final_state 