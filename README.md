# Volt.ai

This repository also holds the **Volt Library** website, a searchable directory of cybersecurity, privacy and AI resources, in [`library/`](library/). See its README to run it.

*Volt.ai* is an AI-powered assistant providing instant answers on privacy, cybersecurity, and AI law.  
It uses Legal-BERT and vector search to deliver fast, accurate legal insights with references to official regulations.

## Features
- Search legal texts (GDPR, NIS2, EU AI Act)
- AI-powered Q&A using Legal-BERT
- Simple web UI with Streamlit

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### Installation
```bash
git clone https://github.com/yourusername/volt-ai.git
cd volt-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
