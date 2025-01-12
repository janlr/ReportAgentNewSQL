import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from agents.insight_generator_agent import InsightGeneratorAgent

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'sales': np.random.normal(1000, 100, 100),
        'customers': np.random.normal(50, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })

@pytest.fixture
def sample_config():
    return {
        "cache_dir": "./test_cache",
        "llm_config": {
            "model": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    }

@pytest.fixture
def agent(sample_config):
    return InsightGeneratorAgent(sample_config)

class TestInsightGeneration:
    """Test cases for generating insights from data."""
    
    def test_generate_basic_insights(self, agent, sample_data):
        """Test generation of basic statistical insights."""
        insights = agent.generate_insights(
            data=sample_data,
            metrics=['sales', 'customers'],
            analysis_type='basic'
        )
        
        assert 'summary' in insights
        assert 'metrics' in insights
        assert len(insights['metrics']) >= 2
        assert any('correlation' in insight.lower() for insight in insights['metrics'])
    
    def test_generate_time_series_insights(self, agent, sample_data):
        """Test generation of time series insights."""
        insights = agent.generate_insights(
            data=sample_data,
            metrics=['sales'],
            dimensions=['date'],
            analysis_type='time_series'
        )
        
        assert 'trend' in insights
        assert 'seasonality' in insights
        assert 'forecast' in insights
    
    def test_generate_correlation_insights(self, agent, sample_data):
        """Test generation of correlation insights."""
        insights = agent.generate_insights(
            data=sample_data,
            metrics=['sales', 'customers'],
            analysis_type='correlation'
        )
        
        assert 'correlations' in insights
        assert len(insights['correlations']) > 0
        assert any('strong correlation' in insight.lower() for insight in insights['correlations'])
    
    def test_generate_anomaly_insights(self, agent, sample_data):
        """Test generation of anomaly detection insights."""
        insights = agent.generate_insights(
            data=sample_data,
            metrics=['sales'],
            analysis_type='anomaly'
        )
        
        assert 'anomalies' in insights
        assert 'threshold' in insights
        assert len(insights['anomalies']) >= 0

class TestInsightCaching:
    """Test cases for insight caching functionality."""
    
    def test_cache_insights(self, agent, sample_data):
        """Test caching of generated insights."""
        metrics = ['sales']
        analysis_type = 'basic'
        
        # Generate insights first time
        insights1 = agent.generate_insights(
            data=sample_data,
            metrics=metrics,
            analysis_type=analysis_type
        )
        
        # Get from cache second time
        insights2 = agent.generate_insights(
            data=sample_data,
            metrics=metrics,
            analysis_type=analysis_type
        )
        
        assert insights1 == insights2
        assert agent.is_cached(metrics, analysis_type)
    
    def test_cache_invalidation(self, agent, sample_data):
        """Test invalidation of cached insights."""
        metrics = ['sales']
        analysis_type = 'basic'
        
        # Generate and cache insights
        agent.generate_insights(
            data=sample_data,
            metrics=metrics,
            analysis_type=analysis_type
        )
        
        # Invalidate cache
        agent.invalidate_cache(metrics, analysis_type)
        assert not agent.is_cached(metrics, analysis_type)

class TestLLMIntegration:
    """Test cases for LLM integration in insight generation."""
    
    @patch('openai.ChatCompletion.create')
    def test_llm_prompt_generation(self, mock_create, agent, sample_data):
        """Test generation of LLM prompts for insights."""
        mock_create.return_value = {
            'choices': [{
                'message': {
                    'content': 'Test insight summary'
                }
            }]
        }
        
        summary = agent.generate_summary(
            data=sample_data,
            insights={'metrics': ['Test insight']},
            report_type='sales'
        )
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        mock_create.assert_called_once()
    
    @patch('openai.ChatCompletion.create')
    def test_llm_error_handling(self, mock_create, agent, sample_data):
        """Test handling of LLM errors during insight generation."""
        mock_create.side_effect = Exception("API Error")
        
        summary = agent.generate_summary(
            data=sample_data,
            insights={'metrics': ['Test insight']},
            report_type='sales'
        )
        
        assert summary.startswith("Error")

class TestDataValidation:
    """Test cases for data validation in insight generation."""
    
    def test_validate_metrics(self, agent, sample_data):
        """Test validation of metric names."""
        with pytest.raises(ValueError):
            agent.generate_insights(
                data=sample_data,
                metrics=['invalid_metric'],
                analysis_type='basic'
            )
    
    def test_validate_analysis_type(self, agent, sample_data):
        """Test validation of analysis type."""
        with pytest.raises(ValueError):
            agent.generate_insights(
                data=sample_data,
                metrics=['sales'],
                analysis_type='invalid_type'
            )
    
    def test_validate_data_structure(self, agent):
        """Test validation of input data structure."""
        invalid_data = pd.DataFrame({'A': []})  # Empty DataFrame
        with pytest.raises(ValueError):
            agent.generate_insights(
                data=invalid_data,
                metrics=['A'],
                analysis_type='basic'
            )

class TestInsightFormatting:
    """Test cases for insight formatting and presentation."""
    
    def test_format_statistical_insights(self, agent, sample_data):
        """Test formatting of statistical insights."""
        insights = agent.generate_insights(
            data=sample_data,
            metrics=['sales'],
            analysis_type='basic'
        )
        
        formatted = agent.format_insights(insights)
        assert isinstance(formatted, dict)
        assert 'summary' in formatted
        assert 'details' in formatted
    
    def test_format_recommendations(self, agent, sample_data):
        """Test formatting of recommendations."""
        insights = agent.generate_insights(
            data=sample_data,
            metrics=['sales', 'customers'],
            analysis_type='correlation'
        )
        
        recommendations = agent.generate_recommendations(insights)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(r, str) for r in recommendations)

if __name__ == "__main__":
    pytest.main([__file__]) 