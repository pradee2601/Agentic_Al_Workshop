# Competitor Differentiation Mapper

A powerful tool that helps founders craft clear market positioning by leveraging AI to analyze competitors and generate differentiation strategies.

## Features

- 🔍 **Competitor Discovery**: Automatically finds and analyzes competitors across multiple platforms
- 📊 **Feature Matrix**: Builds comprehensive feature matrices to compare offerings
- 🎯 **Differentiation Strategy**: Identifies unique opportunities and strategic positioning
- 📈 **Visual Gap Mapping**: Generates interactive visualizations of market gaps and opportunities
- 📥 **Export Analysis**: Download complete analysis reports in JSON format

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd competitor-differentiation-mapper
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run main.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter your startup idea in the text area and click "Analyze Market"

4. View the analysis results, including:
   - Competitor analysis
   - Feature matrix
   - Differentiation strategy
   - Market visualizations

5. Download the complete analysis report using the export button

## Project Structure

```
.
├── agents/
│   ├── competitor_discovery_agent.py
│   ├── feature_matrix_builder_agent.py
│   ├── differentiation_strategist_agent.py
│   └── visual_gap_mapper_agent.py
├── main.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- Google Gemini API
- Tavily Search API
- Other dependencies listed in requirements.txt

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 