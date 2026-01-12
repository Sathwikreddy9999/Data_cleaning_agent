# Streamlit Data Cleaning Agent

A powerful, AI-assisted data cleaning application built with Streamlit.

## Features
- **Auto-Profiling**: Instantly see metadata, missing values, and column types.
- **Data Health Summary**: AI-generated summary of your dataset's quality.
- **Advanced Cleaning**: Manual tools for one-hot encoding, imputation, duplicate removal, and outlier handling.
- **AI Chat Interface**: Chat with your data to request specific cleaning actions (e.g., "Clean missing values in Age").

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   streamlit run app.py
   ```

## Configuration
The app uses `langchain-openai` with OpenRouter. ensuring your `OPENROUTER_API_KEY` is set in your environment variables is recommended for security, though a default key is configured for demo purposes.
