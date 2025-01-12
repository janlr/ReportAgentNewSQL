import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sqlite3
import json

from agents.master_orchestrator_agent import MasterOrchestratorAgent
from agents.database_agent import DatabaseAgent
from agents.insight_generator_agent import InsightGeneratorAgent
from .test_data_generator import TestDataGenerator

@pytest.fixture
def data_generator():
    return TestDataGenerator()

@pytest.fixture
def sample_config():
    return {
        "cache_dir": "./test_cache",
        "database": {
            "host": "localhost",
            "port": 1433,
            "database": "test_db",
            "user": "test_user",
            "password": "test_pass"
        },
        "llm_config": {
            "model": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    }

@pytest.fixture
def orchestrator(sample_config):
    return MasterOrchestratorAgent(sample_config)

class TestErrorRecovery:
    """Test error recovery scenarios in agent interactions."""
    
    @patch.object(DatabaseAgent, 'execute_query')
    def test_database_connection_retry(self, mock_execute_query, orchestrator):
        """Test recovery from temporary database connection issues."""
        # Simulate connection failure then success
        mock_execute_query.side_effect = [
            sqlite3.OperationalError("Connection lost"),
            pd.DataFrame({'sales': [100, 200, 300]})
        ]
        
        result = orchestrator.process({
            'action': 'generate_report',
            'report_type': 'sales'
        })
        
        assert 'report' in result
        assert mock_execute_query.call_count == 2
        assert 'retry_attempts' in result
    
    @patch.object(InsightGeneratorAgent, 'process')
    def test_llm_error_recovery(self, mock_process, orchestrator):
        """Test recovery from LLM API errors."""
        # Simulate LLM API error then success
        mock_process.side_effect = [
            Exception("API rate limit exceeded"),
            {'insights': ['Insight 1', 'Insight 2']}
        ]
        
        result = orchestrator.process({
            'action': 'generate_insights',
            'data': pd.DataFrame({'sales': [100, 200, 300]})
        })
        
        assert 'insights' in result
        assert mock_process.call_count == 2
        assert 'error_recovery' in result
    
    def test_partial_data_recovery(self, orchestrator, data_generator):
        """Test handling of partial data availability."""
        data = data_generator.generate_sales_data()
        # Simulate partial data corruption
        data.loc[10:20, :] = np.nan
        
        result = orchestrator.process({
            'action': 'analyze_data',
            'data': data,
            'metrics': ['sales']
        })
        
        assert 'partial_analysis' in result
        assert 'data_quality_warning' in result
        assert result['analysis_coverage'] > 80  # At least 80% data analyzed

class TestEdgeCaseHandling:
    """Test handling of edge cases in agent interactions."""
    
    def test_empty_result_handling(self, orchestrator, data_generator):
        """Test handling of empty query results."""
        data = pd.DataFrame(columns=['date', 'sales', 'category'])
        
        result = orchestrator.process({
            'action': 'analyze_data',
            'data': data,
            'metrics': ['sales']
        })
        
        assert 'error' in result
        assert result['error']['type'] == 'empty_data'
        assert 'recommendations' in result
    
    def test_invalid_configuration_handling(self, orchestrator):
        """Test handling of invalid agent configurations."""
        # Attempt to process with invalid config
        result = orchestrator.process({
            'action': 'generate_report',
            'invalid_param': 'value',
            'undefined_setting': True
        })
        
        assert 'error' in result
        assert result['error']['type'] == 'invalid_configuration'
        assert 'valid_params' in result
    
    def test_timeout_handling(self, orchestrator, data_generator):
        """Test handling of operation timeouts."""
        # Generate large dataset to force timeout
        data = data_generator.generate_sales_data(periods=100000)
        
        result = orchestrator.process({
            'action': 'analyze_data',
            'data': data,
            'timeout': 1  # Set very short timeout
        })
        
        assert 'timeout_error' in result
        assert 'partial_results' in result
    
    def test_concurrent_modification_handling(self, orchestrator, data_generator):
        """Test handling of concurrent modifications to shared resources."""
        data = data_generator.generate_sales_data()
        
        import threading
        
        def modify_data():
            nonlocal data
            data.loc[0, 'sales'] *= 2
        
        # Create multiple threads to simulate concurrent modifications
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=modify_data)
            thread.start()
            threads.append(thread)
        
        result = orchestrator.process({
            'action': 'analyze_data',
            'data': data,
            'metrics': ['sales']
        })
        
        for thread in threads:
            thread.join()
        
        assert 'concurrent_modification_handled' in result
        assert 'data_version' in result

class TestResourceManagement:
    """Test resource management and cleanup in agent interactions."""
    
    def test_memory_cleanup(self, orchestrator, data_generator):
        """Test proper cleanup of memory resources."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple large datasets
        for _ in range(5):
            data = data_generator.generate_sales_data(periods=10000)
            result = orchestrator.process({
                'action': 'analyze_data',
                'data': data,
                'metrics': ['sales']
            })
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert memory_increase < 500  # Memory increase should be reasonable
        assert 'resources_cleaned' in result
    
    def test_connection_cleanup(self, orchestrator):
        """Test proper cleanup of database connections."""
        # Track initial connection count
        initial_connections = orchestrator.get_active_connections()
        
        # Perform multiple operations
        for _ in range(5):
            result = orchestrator.process({
                'action': 'generate_report',
                'report_type': 'sales'
            })
        
        final_connections = orchestrator.get_active_connections()
        
        assert final_connections <= initial_connections
        assert 'connections_cleaned' in result

if __name__ == "__main__":
    pytest.main([__file__]) 