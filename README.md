# ReportAgentNewSQL

An intelligent reporting system powered by AutoGen that generates insights and reports from SQL databases. Currently demonstrated using AdventureWorks as a sample database, but designed to work with any SQL database.

## Features

- Multi-agent architecture for intelligent report generation
- Support for multiple database types (SQL Server, MySQL, PostgreSQL, SQLite)
- Dynamic data analysis and visualization
- Natural language query processing
- Automated insight generation
- Interactive report generation
- Configurable agent behaviors
- Schema exploration and relationship detection
- AI-powered data analysis and recommendations

## Tech Stack

- **Framework**: AutoGen for multi-agent orchestration
- **Database**: Support for multiple SQL databases (SQL Server, MySQL, PostgreSQL, SQLite)
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy, SQLAlchemy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **AI/ML**: AutoGen framework, Claude API
- **Testing**: pytest, mypy

## Prerequisites

- Python 3.9+
- SQL Database (currently tested with SQL Server)
- Appropriate database drivers (e.g., ODBC Driver 17 for SQL Server)
- Anthropic API key (for AI features)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/janlr/ReportAgentNewSQL.git
   cd ReportAgentNewSQL
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and configure your environment variables:
   ```bash
   cp .env.example .env
   ```

5. Configure your database connection and API keys in `.env`

## Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

- `/agents`: Multi-agent system components
  - `master_orchestrator_agent.py`: Coordinates all other agents
  - `database_agent.py`: Handles database connections and queries
  - `insight_generator_agent.py`: Generates insights from data
  - `report_generator_agent.py`: Creates reports
  - `visualization_agent.py`: Handles data visualization
  - `llm_manager_agent.py`: Manages LLM interactions
- `/utils`: Utility functions and helpers
- `/tests`: Test suite
- `/docs`: Documentation
- `/templates`: Report templates
- `/cache`: Cached data and results
- `/reports`: Generated reports

## Development

1. Install development dependencies:
   ```bash
   pip install -r dev-requirements.txt
   ```

2. Run tests:
   ```bash
   python run_tests.py
   ```

3. Check code quality:
   ```bash
   flake8
   mypy .
   ```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AutoGen framework by Microsoft
- Anthropic's Claude API
- AdventureWorks sample database used for demonstrations 