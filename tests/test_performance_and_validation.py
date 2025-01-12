import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

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

class TestPerformanceBenchmarks:
    """Performance benchmarking tests for various operations."""
    
    def test_large_dataset_processing(self, orchestrator, data_generator):
        """Test processing performance with large datasets."""
        # Generate large dataset
        data = data_generator.generate_sales_data(periods=10000)
        
        start_time = time.time()
        result = orchestrator.process({
            'action': 'analyze_data',
            'data': data,
            'metrics': ['sales']
        })
        processing_time = time.time() - start_time
        
        assert processing_time < 30  # Should process within 30 seconds
        assert 'performance_metrics' in result
    
    def test_concurrent_requests(self, orchestrator, data_generator):
        """Test handling of concurrent report generation requests."""
        data = data_generator.generate_sales_data()
        
        import concurrent.futures
        
        def generate_report():
            return orchestrator.process({
                'action': 'generate_report',
                'data': data,
                'report_type': 'sales'
            })
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_report) for _ in range(5)]
            results = [f.result() for f in futures]
        
        total_time = time.time() - start_time
        
        assert total_time < 60  # Should handle 5 concurrent requests within 60 seconds
        assert all('report' in result for result in results)
    
    def test_memory_usage(self, orchestrator, data_generator):
        """Test memory usage during large data processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large dataset
        data = data_generator.generate_sales_data(periods=50000)
        result = orchestrator.process({
            'action': 'analyze_data',
            'data': data,
            'metrics': ['sales']
        })
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert memory_increase < 1000  # Memory increase should be less than 1GB
        assert 'memory_metrics' in result

class TestDataValidation:
    """Tests for data validation and quality checks."""
    
    def test_data_type_validation(self, orchestrator, data_generator):
        """Test validation of data types in input data."""
        data = data_generator.generate_sales_data()
        # Introduce incorrect data types
        data.loc[10:15, 'sales'] = 'invalid'
        
        result = orchestrator.process({
            'action': 'validate_data',
            'data': data,
            'schema': {
                'sales': 'float64',
                'date': 'datetime64[ns]',
                'category': 'object'
            }
        })
        
        assert 'validation_errors' in result
        assert 'data_type_issues' in result['validation_errors']
        assert len(result['validation_errors']['data_type_issues']) > 0
    
    def test_value_range_validation(self, orchestrator, data_generator):
        """Test validation of value ranges in numeric columns."""
        data = data_generator.generate_inventory_data()
        # Add out-of-range values
        data.loc[5:10, 'stock_level'] = -100
        
        result = orchestrator.process({
            'action': 'validate_data',
            'data': data,
            'constraints': {
                'stock_level': {'min': 0, 'max': 1000},
                'unit_cost': {'min': 0}
            }
        })
        
        assert 'range_violations' in result
        assert len(result['range_violations']['stock_level']) > 0
    
    def test_relationship_validation(self, orchestrator, data_generator):
        """Test validation of relationships between tables."""
        tables = data_generator.generate_related_tables()
        # Introduce invalid relationships
        tables['orders'].loc[0, 'customer_id'] = 999999
        
        result = orchestrator.process({
            'action': 'validate_relationships',
            'data': tables,
            'relationships': [
                {
                    'table1': 'orders',
                    'table2': 'customers',
                    'key1': 'customer_id',
                    'key2': 'customer_id'
                }
            ]
        })
        
        assert 'relationship_violations' in result
        assert len(result['relationship_violations']) > 0
    
    def test_completeness_check(self, orchestrator, data_generator):
        """Test checking for data completeness."""
        data = data_generator.generate_sales_data()
        # Introduce missing values
        data.loc[10:15, 'sales'] = np.nan
        data.loc[20:25, 'category'] = None
        
        result = orchestrator.process({
            'action': 'check_completeness',
            'data': data,
            'required_fields': ['sales', 'category', 'date']
        })
        
        assert 'completeness_report' in result
        assert result['completeness_report']['sales'] < 100
        assert result['completeness_report']['category'] < 100
        assert result['completeness_report']['date'] == 100

if __name__ == "__main__":
    pytest.main([__file__]) 