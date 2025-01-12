# Development Guide

## Quick Start Guide

This guide will help you get started with the reporting system quickly.

### Prerequisites

- Python 3.9 or higher
- Git
- SQL Server or PostgreSQL database
- OpenAI API key (for LLM features)

### 1. Clone and Setup

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

### 2. Configure Environment

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
```ini
DB_HOST=localhost
DB_NAME=your_database
OPENAI_API_KEY=your_api_key
```

### 3. Run the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The application will be available at http://localhost:8501

### 4. Development Workflow

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure code quality:
```bash
# Format code
black .

# Sort imports
isort .

# Run linting
flake8

# Run type checking
mypy .

# Run tests
pytest
```

3. Commit your changes:
```bash
git add .
git commit -m "Add your feature description"
git push origin feature/your-feature-name
```

### 5. Common Tasks

#### Generate a Report
```python
from agents import ReportGeneratorAgent

agent = ReportGeneratorAgent()
report = agent.process({
    'action': 'generate_report',
    'report_type': 'sales',
    'parameters': {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31'
    }
})
```

#### Create a Visualization
```python
from agents import VisualizationAgent

agent = VisualizationAgent()
chart = agent.process({
    'action': 'create_visualization',
    'type': 'line',
    'data': your_data,
    'title': 'Sales Trend'
})
```

### 6. Troubleshooting

1. Database Connection Issues:
   - Check your database credentials in `.env`
   - Ensure the database server is running
   - Verify network connectivity

2. Virtual Environment Issues:
   - Ensure you're in the correct directory
   - Check if the virtual environment is activated
   - Try recreating the virtual environment

3. Import Errors:
   - Verify all dependencies are installed
   - Check your Python path
   - Ensure you're using the correct Python interpreter

For more detailed information, refer to the sections below.

[Rest of the development guide...] 