# AutoGen-Powered Reporting Agent System

A powerful multi-agent system for automated report generation, data analysis, and visualization using AutoGen framework.

## Features

- **Multi-Agent Architecture**: Orchestrated system of specialized agents for different aspects of report generation
- **Intelligent Data Discovery**: Automated database exploration and relationship mapping
- **Dynamic Visualization**: Support for multiple visualization libraries and interactive dashboards
- **Flexible Report Generation**: Multiple output formats including HTML, PDF, and Jupyter notebooks
- **Interactive Dashboards**: Streamlit-based web dashboards with real-time filtering
- **Email Integration**: Automated report delivery via email
- **Extensible Design**: Plugin-based architecture for adding new data sources and visualization types

## System Architecture

### Core Agents

1. **Master Orchestrator Agent**
   - Workflow orchestration and monitoring
   - Agent task delegation
   - Error handling and recovery

2. **User Interface Agent**
   - Natural language request processing
   - Requirement clarification
   - Progress updates

3. **Data Discovery Agent**
   - Database exploration
   - Schema analysis
   - Relationship mapping

4. **Visualization Engine Agent**
   - Dynamic visualization generation
   - Multiple library support
   - Interactive dashboard creation

5. **Report Generation Agent**
   - Template management
   - Multiple output formats
   - Email delivery

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autogen-reporting-system.git
cd autogen-reporting-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Basic Report Generation

```python
from agents.orchestrator import ReportingOrchestrator

# Initialize the orchestrator
orchestrator = ReportingOrchestrator()

# Process a report request
result = orchestrator.process_request(
    "Generate a sales report for Q4 2023 with trend analysis"
)
```

### Custom Visualization

```python
from agents.visualization import VisualizationAgent
import pandas as pd

# Initialize the visualization agent
viz_agent = VisualizationAgent()

# Create a visualization
viz = viz_agent.create_visualization(
    data=your_dataframe,
    viz_type="line",
    config={
        "x": "date",
        "y": "sales",
        "color": "region",
        "title": "Sales Trends by Region"
    }
)

# Save the visualization
viz_agent.save_visualization(viz, "sales_trend")
```

### Interactive Dashboard

```python
from agents.report_generator import ReportGeneratorAgent

# Initialize the report generator
report_agent = ReportGeneratorAgent()

# Create an interactive dashboard
report_agent.create_streamlit_dashboard(
    title="Sales Dashboard",
    data=your_data,
    visualizations=your_visualizations,
    insights=your_insights
)
```

## Configuration

### Database Connection

Edit `.env` file:
```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
```

### Report Templates

Place your report templates in the `templates` directory:
```
templates/
├── report_template.html
├── email_template.html
└── dashboard_template.html
```

## Development

### Running Tests

```bash
pytest tests/
```

### Type Checking

```bash
mypy agents/
```

### Code Formatting

```bash
black agents/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AutoGen framework developers
- Contributors to the various visualization libraries
- The open-source community

## Support

For support, please:
1. Check the [documentation](docs/)
2. Open an issue
3. Contact the maintainers 