import os
import shutil
import json
from pathlib import Path
import logging

def setup_logging():
    """Configure logging for test environment setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_directories():
    """Create necessary directories for testing"""
    directories = [
        'tests',
        'reports',
        'reports/coverage',
        'logs',
        'cache',
        'tests/data',
        'tests/fixtures',
        'reports/visualizations',
        'client_configs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        Path(directory) / '.gitkeep'
        logging.info(f"Created directory: {directory}")

def create_test_config():
    """Create test configuration file"""
    test_config = {
        "database": {
            "test_db": {
                "driver": "sqlite",
                "database": ":memory:"
            }
        },
        "llm": {
            "provider": "mock",
            "mock_responses": "tests/fixtures/mock_llm_responses.json"
        },
        "cache": {
            "type": "memory",
            "ttl": 300
        },
        "logging": {
            "level": "DEBUG",
            "file": "logs/test.log"
        }
    }
    
    with open('tests/test_config.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    logging.info("Created test configuration file")

def create_mock_data():
    """Create mock data and fixtures"""
    mock_llm_responses = {
        "generate_insights": {
            "success": {
                "insights": ["Test insight 1", "Test insight 2"],
                "recommendations": ["Test recommendation 1"]
            },
            "error": {
                "error": "API rate limit exceeded"
            }
        }
    }
    
    with open('tests/fixtures/mock_llm_responses.json', 'w') as f:
        json.dump(mock_llm_responses, f, indent=2)
    logging.info("Created mock LLM responses")

def cleanup_previous():
    """Clean up previous test artifacts"""
    cleanup_dirs = ['reports', 'logs', 'cache']
    for directory in cleanup_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logging.info(f"Cleaned up directory: {directory}")

def main():
    """Main setup function"""
    setup_logging()
    logging.info("Starting test environment setup")
    
    cleanup_previous()
    create_directories()
    create_test_config()
    create_mock_data()
    
    logging.info("Test environment setup completed successfully")

if __name__ == "__main__":
    main() 