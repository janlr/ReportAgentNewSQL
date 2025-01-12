# Quick Start Guide

This guide will help you get started with the reporting system quickly.

## Prerequisites

- Python 3.9 or higher
- Git
- SQL Server or PostgreSQL database
- OpenAI API key (for LLM features)

## Quick Setup Steps

1. **Clone and Setup**
```bash
# Clone the repository
git clone https://github.com/yourusername/reporting-system.git
cd reporting-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run setup script
python setup_dev_env.py
```

2. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Required settings:
DB_HOST=localhost
DB_NAME=your_database
OPENAI_API_KEY=your_api_key
```

3. **Run the Application**
```bash
# Start the Streamlit app
streamlit run app.py
```

The application will be available at http://localhost:8501

## Next Steps

- Read the [Configuration Guide](./configuration.md) for detailed setup options
- Check the [Development Workflow](./workflow.md) for coding guidelines
- Review [Common Tasks](./common_tasks.md) for everyday operations 