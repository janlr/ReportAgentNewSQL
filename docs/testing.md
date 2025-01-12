# Testing Guide

This guide covers testing practices, tools, and procedures for the project.

## Test Environment Setup

1. **Install Test Dependencies**
```bash
pip install -r requirements-test.txt
```

2. **Configure Test Environment**
```bash
# Copy test environment template
cp .env.test.example .env.test

# Edit test settings
TEST_DB_HOST=localhost
TEST_DB_NAME=test_db
```

## Running Tests

### Run All Tests
```bash
# Run with pytest
pytest

# Run with coverage
pytest --cov=agents --cov-report=html
```

### Run Specific Tests
```bash
# Run a specific test file
pytest tests/test_database_agent.py

# Run tests by marker
pytest -m "unit"
pytest -m "integration"
pytest -m "performance"

# Run tests by keyword
pytest -k "database"
```

## Test Categories

### Unit Tests
```python
# Example unit test
def test_report_generation():
    agent = ReportGeneratorAgent()
    report = agent.generate_report(
        report_type="sales",
        parameters={"date": "2023-01-01"}
    )
    assert report is not None
    assert "sales_data" in report
```

### Integration Tests
```python
# Example integration test
@pytest.mark.integration
def test_end_to_end_report():
    # Setup
    db_agent = DatabaseAgent()
    report_agent = ReportGeneratorAgent()
    viz_agent = VisualizationAgent()
    
    # Process
    data = db_agent.get_sales_data()
    report = report_agent.generate_report(data)
    viz = viz_agent.create_visualization(report)
    
    # Assert
    assert data is not None
    assert report["status"] == "success"
    assert viz["type"] == "chart"
```

### Performance Tests
```python
# Example performance test
@pytest.mark.performance
def test_query_performance():
    agent = DatabaseAgent()
    start_time = time.time()
    result = agent.execute_query("SELECT * FROM large_table")
    duration = time.time() - start_time
    
    assert duration < 1.0  # Should complete within 1 second
    assert len(result) > 1000
```

## Test Data Management

### Using Test Fixtures
```python
@pytest.fixture
def sample_data():
    return {
        "sales": [
            {"date": "2023-01-01", "amount": 100},
            {"date": "2023-01-02", "amount": 200}
        ]
    }

def test_with_fixture(sample_data):
    agent = ReportGeneratorAgent()
    report = agent.process_data(sample_data)
    assert report["total"] == 300
```

### Mock Objects
```python
@pytest.mark.unit
def test_with_mocks(mocker):
    # Mock database calls
    mock_db = mocker.patch('agents.DatabaseAgent')
    mock_db.get_data.return_value = sample_data
    
    agent = ReportGeneratorAgent(db=mock_db)
    result = agent.generate_report()
    
    assert mock_db.get_data.called
    assert result is not None
```

## Test Coverage

### Coverage Configuration
```ini
# pytest.ini
[pytest]
addopts = --cov=agents --cov-report=html --cov-report=term-missing

[coverage:run]
source = agents
omit = tests/*,*/__init__.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

### Coverage Reports
```bash
# Generate coverage report
pytest --cov=agents --cov-report=html

# View report
open htmlcov/index.html
```

## Continuous Integration

### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: pytest
```

## Best Practices

1. **Test Organization**
   - Group related tests in classes
   - Use descriptive test names
   - Follow AAA pattern (Arrange, Act, Assert)

2. **Test Data**
   - Use fixtures for common data
   - Clean up test data after tests
   - Use realistic test data

3. **Assertions**
   - Use specific assertions
   - Test both positive and negative cases
   - Include error messages

4. **Documentation**
   - Document test purpose
   - Include example usage
   - Explain complex test setups

For more information about specific test cases, see the [Test Case Reference](./api/test-cases.md). 