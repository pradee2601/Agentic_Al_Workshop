# Competitor Differentiation Mapper

A Streamlit application that helps founders analyze their market position by identifying competitors, analyzing features, and generating differentiation strategies.

## Features

- 🔍 Competitor identification and analysis
- 📊 Feature comparison matrix
- 🎯 Differentiation strategy generation
- 📈 Visual feature gap analysis

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Usage

1. Enter your startup idea in the text input field
2. Click "Analyze Competitors"
3. View the results in three tabs:
   - Competitors: Detailed information about each competitor
   - Differentiation Strategy: Analysis of market gaps and opportunities
   - Feature Map: Visual comparison of features across competitors

## Requirements

- Python 3.8+
- Google API key for Gemini Pro
- Tavily API key for web search 