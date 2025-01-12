import pytest
import sys
import os
from datetime import datetime
import logging

def setup_logging():
    """Configure logging for test execution"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_run_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def run_tests():
    """Run all tests with pytest configuration"""
    log_file = setup_logging()
    logging.info("Starting test execution")
    
    # Configure pytest arguments
    pytest_args = [
        "--verbose",
        "--capture=tee-sys",  # Capture stdout/stderr but also show them
        "--tb=short",         # Shorter traceback format
        "--strict-markers",   # Ensure all markers are registered
        "--durations=10",     # Show 10 slowest tests
        "--maxfail=5",        # Stop after 5 failures
        f"--html=reports/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        "--self-contained-html",
        "--cov=agents",       # Enable coverage for agents directory
        "--cov-report=html:reports/coverage",  # Generate HTML coverage report
        "--cov-report=term-missing",  # Show missing lines in terminal
        "tests/"             # Test directory to scan
    ]
    
    # Run the tests
    logging.info("Running tests with pytest")
    try:
        result = pytest.main(pytest_args)
        
        if result == 0:
            logging.info("All tests passed successfully!")
        else:
            logging.error(f"Tests failed with exit code: {result}")
            
    except Exception as e:
        logging.error(f"Error during test execution: {str(e)}")
        return 1
        
    logging.info(f"Test execution completed. Log file: {log_file}")
    return result

if __name__ == "__main__":
    sys.exit(run_tests()) 