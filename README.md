Velos — AI-Powered Road Fatality Analysis

Advanced traffic safety analytics platform built with Python + Streamlit for exploring, modeling, and interpreting global road fatality datasets.

Features
Data Processing
Supports:
CSV
Excel (.xlsx, .xls)
World Bank exports
Long-format datasets
Automatic:
wide → long reshaping
schema detection
column normalization
missing data cleanup
Interactive Analytics
Country comparison dashboards
Trend visualizations
Time-series exploration
SQL querying using DuckDB
Download filtered datasets
AI-Powered Querying

Ask questions in plain English:

“Which countries improved the most?”
“Show the highest mortality rates”
“What trends exist globally?”

Supports:

Local LLMs via Ollama
Cloud inference via Anthropic Claude
Econometrics Module

Includes:

Pooled OLS
Two-Way Fixed Effects (TWFE)
β-Convergence analysis
Kuznets-style nonlinear trend modeling
Country-wise regression trends
HC3 robust standard errors
Built-In Technologies
Streamlit UI
DuckDB SQL engine
Pandas + NumPy analytics
HTTP AI integrations
Pure Python econometric implementations
Installation

Clone the repository:

git clone https://github.com/yourusername/velos.git
cd velos

Install dependencies:

pip install streamlit pandas numpy duckdb httpx openpyxl
Running the App
streamlit run app.py

Or:

python -m streamlit run app.py
AI Setup
Option 1 — Local AI with Ollama

Install Ollama

Start the server:

ollama serve

Run a model:

ollama run phi3:mini

Then select:

Provider → Ollama (local)
Option 2 — Anthropic Claude

Get an API key from:

Anthropic Console

Enter the key inside the sidebar.

Supported Dataset Structure

Velos automatically detects columns like:

Type	Example
Country	Country Name
Time	Year
Metric	Value, Fatality Rate, Deaths

Works with:

panel datasets
wide time-series exports
country-year mortality tables
Econometric Models
Pooled OLS

Estimates:

log(mortality)=α+β⋅Year+ε

Two-Way Fixed Effects

Controls for:

country fixed effects
year fixed effects

Useful for:

within-country trend estimation
panel analysis
β-Convergence

Tests whether high-mortality countries improve faster:

Δlog(mortality
t
	​

)=α+βlog(mortality
t−1
	​

)+ε

Kuznets Proxy Trend

Quadratic time trend estimation:

log(mortality)=α+β
1
	​

t+β
2
	​

t
2
+ε

Tech Stack
Python
Streamlit
Pandas
NumPy
DuckDB
HTTPX
Ollama
Claude API

Project Structure
velos/
│
├── app.py
├── requirements.txt
├── README.md
└── screenshots/
Example Use Cases
Road safety policy research
Public health analytics
Transportation economics
Global development analysis
Academic econometrics projects
AI-assisted data exploration
Future Improvements
Geographic mapping
Forecasting models
Causal inference modules
PDF export reports
Automated insight generation
Multi-dataset joins
Real-time dashboards
License

MIT License

Author

Built by Siva Harish using Python, econometrics, and AI-assisted analytics.
