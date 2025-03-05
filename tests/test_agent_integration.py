import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio
import time
import os
from agents import initialize_environment

from agents.master_orchestrator_agent import MasterOrchestratorAgent
from agents.database_agent import DatabaseAgent
from agents.insight_generator_agent import InsightGeneratorAgent
from agents.visualization_agent import VisualizationAgent
from agents.report_generator_agent import ReportGeneratorAgent
from agents.user_interface_agent import UserInterfaceAgent
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

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    os.environ['DB_HOST'] = 'test_host'
    os.environ['DB_PORT'] = '1433'
    os.environ['DB_NAME'] = 'test_db'
    os.environ['DB_USER'] = 'test_user'
    os.environ['DB_PASSWORD'] = 'test_password'
    os.environ['OPENAI_API_KEY'] = 'test_openai_key'
    os.environ['ENVIRONMENT'] = 'testing'
    initialize_environment()

class TestAgentCommunicationFailures:
    """Test scenarios where agent communication fails."""
    
    def test_agent_unavailable(self, orchestrator):
        """Test handling when an agent becomes unavailable."""
        with patch.object(InsightGeneratorAgent, 'process', side_effect=ConnectionError):
            result = orchestrator.process({
                'action': 'generate_report',
                'report_type': 'sales',
                'include_insights': True
            })
            
            assert 'error_handled' in result
            assert result['partial_completion']
            assert 'insights' not in result
            assert 'report' in result  # Other components should still work
    
    def test_agent_timeout(self, orchestrator):
        """Test handling of agent response timeouts."""
        async def slow_process(*args, **kwargs):
            await asyncio.sleep(5)
            return {'insights': ['Insight 1']}
            
        with patch.object(InsightGeneratorAgent, 'process', side_effect=slow_process):
            result = orchestrator.process({
                'action': 'generate_report',
                'report_type': 'sales',
                'timeout': 2
            })
            
            assert 'timeout_handled' in result
            assert 'fallback_action_taken' in result

class TestPerformanceDegradation:
    """Test handling of performance degradation scenarios."""
    
    def test_high_latency_handling(self, orchestrator, data_generator):
        """Test system behavior under high latency conditions."""
        data = data_generator.generate_sales_data()
        
        with patch.object(DatabaseAgent, 'execute_query', 
                         side_effect=lambda *args, **kwargs: time.sleep(2) or pd.DataFrame()):
            start_time = time.time()
            result = orchestrator.process({
                'action': 'analyze_data',
                'data': data,
                'metrics': ['sales']
            })
            duration = time.time() - start_time
            
            assert 'performance_warning' in result
            assert result['latency_detected']
            assert 'mitigation_applied' in result
    
    def test_resource_exhaustion(self, orchestrator, data_generator):
        """Test handling of resource exhaustion scenarios."""
        # Generate multiple large datasets
        datasets = [data_generator.generate_sales_data(periods=10000) for _ in range(5)]
        
        results = []
        for data in datasets:
            result = orchestrator.process({
                'action': 'analyze_data',
                'data': data,
                'metrics': ['sales']
            })
            results.append(result)
            
        assert any('resource_warning' in r for r in results)
        assert all('completed' in r for r in results)

class TestDataConsistency:
    """Test data consistency across agent interactions."""
    
    def test_cross_agent_data_consistency(self, orchestrator, data_generator):
        """Test consistency of data across multiple agent interactions."""
        data = data_generator.generate_sales_data()
        
        result = orchestrator.process({
            'action': 'generate_report',
            'data': data,
            'report_type': 'sales',
            'include_insights': True,
            'include_visualizations': True
        })
        
        # Check data consistency across components
        assert result['report']['total_sales'] == result['insights']['analyzed_sales']
        assert result['visualizations']['data_points'] == len(data)
        assert 'data_consistency_verified' in result
    
    def test_concurrent_updates(self, orchestrator, data_generator):
        """Test data consistency during concurrent updates."""
        data = data_generator.generate_sales_data()
        
        async def run_concurrent_updates():
            tasks = []
            for i in range(5):
                task = asyncio.create_task(orchestrator.process_async({
                    'action': 'update_data',
                    'data': data,
                    'update_id': i
                }))
                tasks.append(task)
            return await asyncio.gather(*tasks)
            
        results = asyncio.run(run_concurrent_updates())
        
        assert all('update_successful' in r for r in results)
        assert len(set(r['data_version'] for r in results)) == 1  # All updates should see same version

@pytest.mark.asyncio
async def test_full_report_workflow(orchestrator):
    """Test complete workflow from data to report."""
    result = await orchestrator.process({
        "workflow": "report_generation",
        "action": "generate_report",
        "parameters": {
            "report_type": "Sales Analysis",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
    })
    
    assert result["success"] == True
    assert "data" in result
    assert "charts" in result["data"]
    assert "insights" in result["data"]

if __name__ == "__main__":
    pytest.main([__file__]) 