"""
Multi-Agent Travel Planner System
Assignment: Build a Multi-Agent System using LangChain + LangGraph

This system uses 4 specialized agents to create personalized travel itineraries:
1. Travel Intent Parser - Extracts destination, duration, budget, and interests
2. Destination Researcher - Gathers information about the destination
3. Itinerary Planner - Creates detailed day-by-day itineraries
4. Travel Advisor - Provides travel tips, safety info, and budget advice

The agents collaborate through a shared state (TravelState) and are orchestrated
using LangGraph's directed workflow.

Usage:
    CLI Mode: python multi_agent_system.py
    Streamlit Mode: streamlit run multi_agent_system.py
"""

import os
import sys
import json
from typing import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# SHARED STATE DEFINITION
# ============================================================================

class TravelState(TypedDict):
    """
    Shared state passed between all agents in the workflow.
    Each agent reads from and writes to specific fields.
    """
    user_input: str          # Raw free-text from the user
    destination: str         # Extracted by Intent Parser
    duration: str            # Extracted by Intent Parser
    budget: str             # Extracted by Intent Parser
    interests: str           # Extracted by Intent Parser
    research_notes: str      # Populated by Destination Researcher
    itinerary: str           # Populated by Itinerary Planner
    travel_tips: str         # Populated by Travel Advisor


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Get API key from environment or Streamlit secrets
def get_api_key():
    """Get Groq API key from environment variables or Streamlit secrets"""
    # Try to get from environment first (local development)
    api_key = os.getenv("GROQ_API_KEY")
    
    # If not found, try Streamlit secrets (cloud deployment)
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except:
            pass
    
    return api_key

# Shared LLM instance — API key is read from GROQ_API_KEY env var or Streamlit secrets
# Using Groq's free tier with llama-3.3-70b-versatile model
api_key = get_api_key()
if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in:\n"
        "- Local: .env file with GROQ_API_KEY=your_key\n"
        "- Streamlit Cloud: App Settings → Secrets → Add GROQ_API_KEY = \"your_key\""
    )

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, api_key=api_key)


# ============================================================================
# AGENT 1: TRAVEL INTENT PARSER AGENT
# ============================================================================

def travel_intent_parser_node(state: TravelState) -> dict:
    """
    Travel Intent Parser Agent: Extracts trip details from user input.
    
    Input: user_input from TravelState
    Output: Updates destination, duration, budget, interests in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel planning assistant. Extract structured travel intent from the user's request.\n"
         "Return ONLY a valid JSON object with these exact keys: destination, duration, budget, interests.\n"
         "Do not include any other text, explanations, or markdown formatting.\n"
         "Example output:\n"
         '{{"destination": "Tokyo, Japan", "duration": "5 days", "budget": "mid-range", "interests": "anime, sushi, history"}}'),
        ("human", "Travel request: {user_input}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"user_input": state["user_input"]})
    
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        parsed = json.loads(content)
        return {
            "destination": parsed.get("destination", ""),
            "duration": parsed.get("duration", ""),
            "budget": parsed.get("budget", ""),
            "interests": parsed.get("interests", ""),
        }
    except Exception as e:
        print(f"Warning: Failed to parse intent response: {e}")
        return {"destination": "", "duration": "", "budget": "", "interests": ""}


# ============================================================================
# AGENT 2: DESTINATION RESEARCHER AGENT
# ============================================================================

def destination_researcher_node(state: TravelState) -> dict:
    """
    Destination Researcher Agent: Gathers info about the destination.
    
    Input: destination, interests from TravelState
    Output: Updates research_notes in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a destination research specialist. Provide information about attractions, weather, \n"
         "local culture, and hidden gems based on the destination and user interests."),
        ("human", "Destination: {destination}\nInterests: {interests}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "interests": state["interests"],
    })
    return {"research_notes": response.content}


# ============================================================================
# AGENT 3: ITINERARY PLANNER AGENT
# ============================================================================

def itinerary_planner_node(state: TravelState) -> dict:
    """
    Itinerary Planner Agent: Creates detailed day-by-day itineraries.
    
    Input: destination, duration, budget, interests, research_notes from TravelState
    Output: Updates itinerary in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert travel itinerary planner. Create a detailed day-by-day itinerary including:\n"
         "- Activities and sights to visit\n"
         "- Suggested timing for each day\n"
         "- Logistics and transport suggestions\n"
         "Base it on the destination, duration, budget, interests, and research notes provided."),
        ("human",
         "Destination: {destination}\n"
         "Duration: {duration}\n"
         "Budget: {budget}\n"
         "Interests: {interests}\n"
         "Research Notes: {research_notes}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "duration": state["duration"],
        "budget": state["budget"],
        "interests": state["interests"],
        "research_notes": state["research_notes"],
    })
    return {"itinerary": response.content}


# ============================================================================
# AGENT 4: TRAVEL ADVISOR AGENT
# ============================================================================

def travel_advisor_node(state: TravelState) -> dict:
    """
    Travel Advisor Agent: Provides travel tips, safety info, and budget advice.
    
    Input: destination, budget, itinerary from TravelState
    Output: Updates travel_tips in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional travel advisor. Provide essential advice including:\n"
         "- Safety tips and local laws\n"
         "- Cultural etiquette and basic phrases\n"
         "- Budget management and cost-saving tips for the specific destination"),
        ("human",
         "Destination: {destination}\n"
         "Budget: {budget}\n"
         "Itinerary: {itinerary}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "budget": state["budget"],
        "itinerary": state["itinerary"],
    })
    return {"travel_tips": response.content}


# ============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# ============================================================================

def build_graph():
    """
    Build the LangGraph workflow connecting all agents.
    
    Workflow:
        START → Intent Parser → Destination Researcher → Itinerary Planner → Travel Advisor → END
    """
    graph = StateGraph(TravelState)
    
    # Add nodes (agents)
    graph.add_node("intent_parser", travel_intent_parser_node)
    graph.add_node("destination_researcher", destination_researcher_node)
    graph.add_node("itinerary_planner", itinerary_planner_node)
    graph.add_node("travel_advisor", travel_advisor_node)
    
    # Define workflow edges
    graph.set_entry_point("intent_parser")
    graph.add_edge("intent_parser", "destination_researcher")
    graph.add_edge("destination_researcher", "itinerary_planner")
    graph.add_edge("itinerary_planner", "travel_advisor")
    graph.add_edge("travel_advisor", END)
    
    return graph.compile()


# ============================================================================
# EXECUTION FUNCTION
# ============================================================================

def execute_travel_planning(user_input: str) -> TravelState:
    """
    Execute the multi-agent travel planning workflow.
    
    Args:
        user_input: User's travel request in natural language
    
    Returns:
        TravelState: Final state with all fields populated by agents
    """
    initial_state: TravelState = {
        "user_input": user_input,
        "destination": "",
        "duration": "",
        "budget": "",
        "interests": "",
        "research_notes": "",
        "itinerary": "",
        "travel_tips": "",
    }
    
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return final_state


# ============================================================================
# CLI MODE
# ============================================================================

def print_itinerary(state: TravelState) -> None:
    """Print itinerary to console (CLI mode)"""
    print("\n" + "="*80)
    print("TRAVEL PLAN GENERATED")
    print("="*80)
    
    print(f"\n[DESTINATION]: {state['destination']}")
    print(f"[DURATION]: {state['duration']}")
    print(f"[BUDGET]: {state['budget']}")
    print(f"[INTERESTS]: {state['interests']}")
    
    print("\n[ITINERARY]")
    print("-" * 80)
    print(state["itinerary"] if state["itinerary"] else "No itinerary generated")
    
    print("\n[TRAVEL TIPS & ADVICE]")
    print("-" * 80)
    print(state["travel_tips"] if state["travel_tips"] else "No travel tips generated")
    
    print("\n" + "="*80)


def main() -> None:
    """Main function for CLI mode"""
    print("="*80)
    print("MULTI-AGENT TRAVEL PLANNER")
    print("="*80)
    print("\nThis system uses 4 AI agents to plan your perfect trip:")
    print("  1. Travel Intent Parser - Extracts destination, duration, budget, and interests")
    print("  2. Destination Researcher - Finds attractions and local info")
    print("  3. Itinerary Planner - Creates detailed day-by-day itineraries")
    print("  4. Travel Advisor - Provides safety, cultural, and budget tips")
    print("\n" + "="*80 + "\n")
    
    # Get user input
    user_input = ""
    while not user_input.strip():
        user_input = input("Enter your travel request (e.g., '5 days in Paris on a budget'): ")
    
    print("\nProcessing: Planning your trip through 4 specialized agents...")
    print("   This may take 15-30 seconds...\n")
    
    # Execute workflow
    final_state = execute_travel_planning(user_input)
    
    # Display results
    print_itinerary(final_state)


# ============================================================================
# STREAMLIT MODE
# ============================================================================

def run_streamlit_app():
    """Run Streamlit web interface"""
    import streamlit as st
    
    # Page configuration
    st.set_page_config(
        page_title="Multi-Agent Travel Planner",
        page_icon="✈️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .agent-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .result-section {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1E88E5;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">✈️ Multi-Agent Travel Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by LangChain + LangGraph | 4 Specialized AI Agents</div>', unsafe_allow_html=True)
    
    # Sidebar - Agent Information
    with st.sidebar:
        st.header("🤖 Agent System")
        st.markdown("---")
        
        st.markdown("### Agent 1: Travel Intent Parser")
        st.markdown("📋 Extracts destination, duration, budget, and interests")
        
        st.markdown("### Agent 2: Destination Researcher")
        st.markdown("🔍 Finds attractions and local info")
        
        st.markdown("### Agent 3: Itinerary Planner")
        st.markdown("🗺️ Creates detailed day-by-day itineraries")
        
        st.markdown("### Agent 4: Travel Advisor")
        st.markdown("💡 Provides safety, cultural, and budget tips")
        
        st.markdown("---")
        st.markdown("**Technology Stack:**")
        st.markdown("- LangGraph for workflow")
        st.markdown("- LangChain for LLM integration")
        st.markdown("- Groq API (llama-3.3-70b)")
        st.markdown("- Streamlit for UI")
    
    # Main content
    st.markdown("### 📝 Enter Your Travel Request")
    st.markdown("Describe your dream trip in natural language. Include destination, duration, budget, and interests.")
    
    # Example requests
    with st.expander("💡 See Example Requests"):
        st.markdown("""
        - "I want to visit Tokyo for 5 days on a mid-range budget. I love sushi and anime."
        - "Planning a romantic weekend trip to Paris with a focus on art and history."
        - "Looking for a 10-day adventure trip in Bali on a backpacker budget."
        - "Need a luxury itinerary for 1 week in New York City with the best dining spots."
        """)
    
    # User input
    user_input = st.text_area(
        "Your Travel Request:",
        height=100,
        placeholder="Example: I want to visit Tokyo for 5 days on a mid-range budget. I love sushi and anime."
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_button = st.button("🚀 Plan My Trip", use_container_width=True, type="primary")
    
    # Process request
    if generate_button:
        if not user_input.strip():
            st.error("⚠️ Please enter a travel request")
        else:
            # Progress indicator
            with st.spinner("🔄 Planning your trip through 4 AI agents... This may take 15-30 seconds..."):
                try:
                    # Execute workflow
                    final_state = execute_travel_planning(user_input)
                    
                    # Display success
                    st.success("✅ Travel plan generated successfully!")
                    
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📍 Overview", "🗺️ Itinerary", "💡 Advice", "🔍 Research"])
                    
                    with tab1:
                        st.markdown("### 📍 Trip Overview")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Destination:** {final_state['destination']}")
                            st.markdown(f"**Duration:** {final_state['duration']}")
                        
                        with col2:
                            st.markdown(f"**Budget:** {final_state['budget']}")
                            st.markdown(f"**Interests:** {final_state['interests']}")
                    
                    with tab2:
                        st.markdown("### 🗺️ Detailed Itinerary")
                        if final_state["itinerary"]:
                            st.markdown(final_state["itinerary"])
                        else:
                            st.warning("No itinerary generated")
                    
                    with tab3:
                        st.markdown("### 💡 Travel Tips & Advice")
                        if final_state["travel_tips"]:
                            st.markdown(final_state["travel_tips"])
                        else:
                            st.warning("No advice generated")
                    
                    with tab4:
                        st.markdown("### 🔍 Destination Research")
                        if final_state["research_notes"]:
                            st.markdown(final_state["research_notes"])
                        else:
                            st.warning("No research notes generated")
                    
                    # Download option
                    st.markdown("---")
                    itinerary_text = f"""
TRAVEL PLAN: {final_state['destination']}
=========================================

Duration: {final_state['duration']}
Budget: {final_state['budget']}
Interests: {final_state['interests']}

Itinerary:
{final_state['itinerary']}

Travel Tips:
{final_state['travel_tips']}
"""

                    st.download_button(
                        label="📥 Download Travel Plan",
                        data=itinerary_text,
                        file_name=f"travel_plan_{final_state['destination'].replace(' ', '_')}.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"An error occurred while planning the trip: {e}")

if __name__ == "__main__":
    main()