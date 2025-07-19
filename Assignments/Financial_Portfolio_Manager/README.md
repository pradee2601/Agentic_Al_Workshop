# Financial Portfolio Manager

An AI-powered financial portfolio analysis tool that provides personalized investment recommendations.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Google API Key

1. **Get a Google API Key:**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

2. **Create a .env file:**
   - Create a file named `.env` in this directory
   - Add the following content:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```
   - Replace `your_actual_api_key_here` with your real API key

### 3. Run the Application
```bash
streamlit run main.py
```

## Features

- Portfolio analysis and strategy recommendation
- Personalized investment recommendations
- Risk assessment
- Implementation planning
- Comprehensive financial reports

## Usage

1. Fill in your financial details (salary, age, expenses, goals)
2. Enter your current portfolio information
3. Select your risk tolerance
4. Click "Generate Report" to get your personalized financial analysis 