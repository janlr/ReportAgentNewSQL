# Test Execution Guide

This guide explains how to run tests for the AutoGen-Powered Reporting System.

## Prerequisites

1. Ensure you have the test dependencies installed:
```bash
pip install -r requirements-test.txt
```

2. Set up the test environment:
```bash
# Copy test environment template
cp .env.test.example .env.test

# Configure test database and other settings in .env.test
TEST_DB_HOST=localhost
TEST_DB_NAME=test_db
TEST_DB_USER=your_test_user
TEST_DB_PASSWORD=your_test_password
```

## Running Tests

### Quick Start
To run all tests:
```bash
pytest
```

### Run with Coverage
To run tests and generate a coverage report:
```bash
# Run with terminal coverage report
pytest --cov=agents

# Generate HTML coverage report
pytest --cov=agents --cov-report=html
# View the report at htmlcov/index.html
```

### Running Specific Tests

#### By Test Type
```bash
# Run only unit tests
pytest -m "unit"

# Run only integration tests
pytest -m "integration"

# Run only performance tests
pytest -m "performance"
```

#### By Module
```bash
# Test specific agent
pytest tests/test_database_agent.py
pytest tests/test_report_generator.py

# Test specific class
pytest tests/test_database_agent.py::TestDatabaseAgent

# Test specific method
pytest tests/test_database_agent.py::TestDatabaseAgent::test_connection
```

#### By Pattern
```bash
# Run tests matching a keyword
pytest -k "database"
pytest -k "report or visualization"
```

### Test Options

#### Verbose Output
```bash
# Show more detailed test output
pytest -v
```

#### Show Print Statements
```bash
# Show print output during tests
pytest -s
```

#### Stop on First Failure
```bash
# Stop execution on first failure
pytest -x
```

#### Parallel Execution
```bash
# Run tests in parallel (4 processes)
pytest -n 4
```

## Debugging Tests

### Debug on Failure
```bash
# Enter debugger on test failure
pytest --pdb
```

### Debug with Print Statements
Add these lines to your test:
```python
import pprint
pprint.pprint(your_variable)  # Pretty print complex objects
```

### Using pytest.set_trace()
Add this line where you want to break:
```python
import pytest; pytest.set_trace()
```

## Test Configuration

### pytest.ini
Key settings in `pytest.ini`:
```ini
[pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

### Coverage Configuration
Settings in `.coveragerc`:
```ini
[run]
source = agents
omit =
    tests/*
    */__init__.py
    */migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

## Continuous Integration

Tests are automatically run on:
- Every push to main branch
- Every pull request
- Nightly builds

GitHub Actions workflow is configured in `.github/workflows/test.yml`

## Common Issues

### Database Connection Issues
- Ensure test database is running
- Verify test database credentials in `.env.test`
- Check database permissions

### Import Errors
- Verify virtual environment is activated
- Check all dependencies are installed
- Ensure PYTHONPATH includes project root

### Test Data Issues
- Check test data fixtures are present
- Verify test database is properly seeded
- Ensure test data is cleaned up after tests

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