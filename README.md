# Multi-Agent Travel Planner System

A sophisticated AI-powered travel planning system that uses 4 specialized agents working together to create personalized travel itineraries. Built with **LangChain**, **LangGraph**, and **Groq API**.

## 🎯 Project Overview

This system demonstrates multi-agent collaboration in action. Instead of a single AI model, four specialized agents work together in a coordinated workflow to plan your perfect trip:

1. **Travel Intent Parser** - Extracts and structures travel requirements from natural language
2. **Destination Researcher** - Gathers comprehensive information about destinations
3. **Itinerary Planner** - Creates detailed day-by-day travel plans
4. **Travel Advisor** - Provides safety, cultural, and budget guidance

## 🤖 The Four Agents

### 1. Travel Intent Parser Agent
- **Purpose**: Extracts structured travel data from user input
- **Extracts**: Destination, duration, budget level, and interests
- **Output Format**: JSON with destination, duration, budget, and interests
- **Example**: "5 days in Paris on a budget" → `{destination: "Paris, France", duration: "5 days", budget: "budget-friendly", interests: "art, history"}`

### 2. Destination Researcher Agent
- **Purpose**: Researches attractions, culture, weather, and hidden gems
- **Input**: Destination and user interests
- **Output**: Comprehensive research notes tailored to user interests
- **Covers**: Local attractions, weather, culture, cuisine, and off-the-beaten-path recommendations

### 3. Itinerary Planner Agent
- **Purpose**: Creates detailed, personalized day-by-day itineraries
- **Input**: Destination, duration, budget, interests, and research notes
- **Output**: Structured itinerary with activities, timing, and logistics
- **Considers**: Budget constraints, user interests, practical logistics, and travel time

### 4. Travel Advisor Agent
- **Purpose**: Provides essential travel guidance
- **Input**: Destination, budget, and generated itinerary
- **Output**: Safety tips, cultural etiquette, local phrases, and budget advice
- **Covers**: Health & safety, local laws, cultural norms, money-saving tips

## 🏗️ Architecture

### Workflow Flow
```
START
  ↓
[Intent Parser] → Extracts trip details
  ↓
[Destination Researcher] → Gathers destination info
  ↓
[Itinerary Planner] → Creates day-by-day plan
  ↓
[Travel Advisor] → Provides tips & advice
  ↓
END
```

### Shared State (TravelState)
All agents share a common state object that passes data between them:

```python
{
    "user_input": str,       # User's raw request
    "destination": str,      # Extracted destination
    "duration": str,         # Trip duration
    "budget": str,          # Budget level
    "interests": str,       # User interests
    "research_notes": str,  # Destination research
    "itinerary": str,       # Day-by-day itinerary
    "travel_tips": str      # Travel advice
}
```

## 🛠️ Technology Stack

- **LangGraph**: Multi-agent workflow orchestration
- **LangChain**: LLM integration and prompt management
- **Groq API**: Fast LLM inference (llama-3.3-70b-versatile)
- **Python 3.8+**: Core language
- **Streamlit**: Web UI (optional)
- **Python dotenv**: Environment variable management

## 📋 Requirements

- Python 3.8 or higher
- Groq API key 

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/JaySoni1/Multi-Agent.git
cd Multi-Agent
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

**Getting a Groq API Key:**
1. Visit https://console.groq.com
2. Sign up for a free account
3. Create an API key
4. Add it to your `.env` file

## 💻 Usage

### CLI Mode (Simple Command Line)
```bash
python agent.py
```

This will:
1. Prompt you to enter a travel request
2. Process through all 4 agents
3. Display the complete travel plan in your terminal

**Example input:**
```
Enter your travel request: I want to visit Tokyo for 5 days on a mid-range budget. I love anime, sushi, and temples.
```

### Streamlit Web UI (Interactive)
```bash
streamlit run agent.py
```

This launches a modern web interface with:
- Beautiful UI with organized tabs
- Real-time processing with status indicators
- Tabbed display for different aspects of the plan
- Download button to save the itinerary as a text file

**Streamlit Features:**
- 📝 Input area for travel requests
- 💡 Example requests for inspiration
- 🔄 Real-time processing indicator
- 📊 Organized tabs for Overview, Itinerary, Advice, and Research
- 📥 Download button for the generated plan

## 📝 Example Usage

### Example 1: Budget Beach Vacation
**Input:**
```
"Planning a 7-day trip to Bali with a backpacker budget. I love surfing, yoga, and local food."
```

**Output includes:**
- Destination: Bali, Indonesia
- Duration: 7 days
- Budget: Backpacker-friendly
- Day-by-day itinerary with beach spots, yoga retreats, and local eateries
- Safety tips for Indonesia
- Budget-saving recommendations

### Example 2: Luxury City Break
**Input:**
```
"Luxury weekend in New York City. Interested in fine dining and Broadway shows."
```

**Output includes:**
- Destination: New York City, USA
- Duration: 2-3 days
- Budget: Luxury
- Curated list of 5-star restaurants and Broadway theater recommendations
- Insider tips for accessing exclusive experiences
- Currency and tipping guides

### Example 3: Adventure Trip
**Input:**
```
"10-day adventure in New Zealand with hiking and natural scenery. Mid-range budget."
```

**Output includes:**
- Day-by-day hiking routes and scenic locations
- Accommodation options at each stop
- Equipment and preparation advice
- Safety considerations for outdoor activities
- Budget-friendly dining and activity options

## 🔧 Configuration

### Customizing the LLM Model
Edit the model selection in `agent.py`:

```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Change model here
    temperature=0.7,                   # Adjust creativity (0-1)
    api_key=api_key
)
```

Available Groq models:
- `llama-3.3-70b-versatile` (Recommended - default)
- `mixtral-8x7b-32768`
- `gemma-7b-it`

### Adjusting Temperature
- **Lower (0-0.5)**: More factual, consistent responses
- **Higher (0.7-1.0)**: More creative, diverse responses
- **Current**: 0.7 (balanced)

## 📂 Project Structure

```
Multi-Agent/
├── agent.py              # Main application (CLI + Streamlit)
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore file
├── requirements.txt     # Project dependencies
└── README.md           # This file
```


**Created with ❤️ using LangChain + LangGraph**
