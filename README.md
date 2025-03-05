# ReportAgentNewSQL

An intelligent reporting system powered by AutoGen that generates insights and reports from SQL databases.

## Features

- Multi-agent architecture for intelligent report generation
- Support for multiple database types (SQL Server, MySQL, PostgreSQL, SQLite)
- Dynamic data analysis and visualization
- Natural language query processing
- Automated insight generation
- Interactive report generation
- Configurable agent behaviors

## Installation

1. Clone the repository:
```bash
git clone https://github.com/janlr/ReportAgentNewSQL.git
cd ReportAgentNewSQL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Configure your database connection in `.env`:
```
DB_TYPE=mssql
DB_HOST=your_host
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password
```

2. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
ReportAgentNewSQL/
├── agents/                 # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py
│   ├── master_orchestrator_agent.py
│   ├── database_agent.py
│   ├── data_manager_agent.py
│   ├── user_interface_agent.py
│   ├── report_generator_agent.py
│   ├── insight_generator_agent.py
│   ├── assistant_agent.py
│   └── llm_manager_agent.py
├── app.py                 # Main application entry point
├── requirements.txt       # Project dependencies
├── .env.example          # Example environment variables
└── README.md             # Project documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [AutoGen](https://github.com/microsoft/autogen)
- Powered by OpenAI and Anthropic LLMs
- Uses Streamlit for the user interface 