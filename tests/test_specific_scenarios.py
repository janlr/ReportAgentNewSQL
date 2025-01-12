import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

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

class TestSeasonalityAnalysis:
    """Test cases for seasonal pattern detection and analysis."""
    
    def test_detect_yearly_seasonality(self, orchestrator, data_generator):
        """Test detection of yearly seasonal patterns."""
        # Generate time series data with strong yearly seasonality
        data = data_generator.generate_time_series_data(
            periods=730,  # 2 years
            seasonality=0.5,  # Strong seasonality
            metrics=['sales']
        )
        
        result = orchestrator.process({
            'action': 'analyze_seasonality',
            'data': data,
            'metric': 'sales'
        })
        
        assert 'seasonal_patterns' in result
        assert 'yearly_pattern' in result['seasonal_patterns']
        assert result['seasonal_patterns']['yearly_pattern']['strength'] > 0.4
    
    def test_detect_weekly_patterns(self, orchestrator, data_generator):
        """Test detection of weekly patterns in sales data."""
        data = data_generator.generate_sales_data(periods=90)  # 3 months
        # Add weekly pattern
        data['sales'] += np.sin(np.arange(90) * 2 * np.pi / 7) * 100
        
        result = orchestrator.process({
            'action': 'analyze_seasonality',
            'data': data,
            'metric': 'sales'
        })
        
        assert 'weekly_pattern' in result['seasonal_patterns']
        assert result['seasonal_patterns']['weekly_pattern']['strength'] > 0.3

class TestAnomalyDetection:
    """Test cases for anomaly detection in different scenarios."""
    
    def test_detect_sales_spikes(self, orchestrator, data_generator):
        """Test detection of unusual sales spikes."""
        data = data_generator.generate_sales_data()
        # Insert anomalies
        data.loc[50:52, 'sales'] *= 3  # Triple sales for 3 days
        
        result = orchestrator.process({
            'action': 'detect_anomalies',
            'data': data,
            'metric': 'sales'
        })
        
        assert 'anomalies' in result
        assert len(result['anomalies']) >= 3
        assert all(50 <= idx <= 52 for idx in result['anomalies']['sales_spikes'])
    
    def test_detect_inventory_anomalies(self, orchestrator, data_generator):
        """Test detection of unusual inventory patterns."""
        data = data_generator.generate_inventory_data()
        # Create some anomalies
        data.loc[10:15, 'stock_level'] = 0  # Stock-outs
        data.loc[20:25, 'stock_level'] = 9999  # Overstocking
        
        result = orchestrator.process({
            'action': 'detect_anomalies',
            'data': data,
            'metrics': ['stock_level']
        })
        
        assert 'stock_outs' in result['anomalies']
        assert 'overstock' in result['anomalies']
        assert len(result['anomalies']['stock_outs']) >= 6
        assert len(result['anomalies']['overstock']) >= 6

class TestCustomerSegmentation:
    """Test cases for customer segmentation analysis."""
    
    def test_rfm_segmentation(self, orchestrator, data_generator):
        """Test RFM (Recency, Frequency, Monetary) segmentation."""
        data = data_generator.generate_customer_data()
        
        result = orchestrator.process({
            'action': 'segment_customers',
            'data': data,
            'method': 'rfm'
        })
        
        assert 'segments' in result
        assert len(result['segments']) >= 3  # At least 3 segments
        assert 'segment_profiles' in result
        assert 'recommendations' in result
    
    def test_behavioral_segmentation(self, orchestrator, data_generator):
        """Test behavioral segmentation based on purchase patterns."""
        tables = data_generator.generate_related_tables()
        
        result = orchestrator.process({
            'action': 'segment_customers',
            'data': tables,
            'method': 'behavioral'
        })
        
        assert 'segments' in result
        assert 'purchase_patterns' in result
        assert 'segment_transitions' in result  # Customer movement between segments

class TestPredictiveAnalysis:
    """Test cases for predictive analysis capabilities."""
    
    def test_sales_forecast(self, orchestrator, data_generator):
        """Test sales forecasting with different components."""
        data = data_generator.generate_time_series_data(
            periods=365,
            metrics=['sales'],
            trend=0.2,
            seasonality=0.3
        )
        
        result = orchestrator.process({
            'action': 'forecast',
            'data': data,
            'target': 'sales',
            'periods': 30  # Forecast next 30 days
        })
        
        assert 'forecast' in result
        assert len(result['forecast']) == 30
        assert 'confidence_intervals' in result
        assert 'components' in result  # Trend, seasonality, residuals
    
    def test_inventory_optimization(self, orchestrator, data_generator):
        """Test inventory optimization predictions."""
        data = data_generator.generate_inventory_data()
        
        result = orchestrator.process({
            'action': 'optimize_inventory',
            'data': data,
            'target': 'stock_level'
        })
        
        assert 'optimal_levels' in result
        assert 'reorder_points' in result
        assert 'cost_savings' in result

class TestEdgeCases:
    """Test cases for handling edge cases and unusual situations."""
    
    def test_handle_missing_data(self, orchestrator, data_generator):
        """Test handling of missing data in time series."""
        data = data_generator.generate_sales_data()
        # Introduce missing values
        data.loc[10:15, 'sales'] = np.nan
        
        result = orchestrator.process({
            'action': 'analyze_data',
            'data': data,
            'metrics': ['sales']
        })
        
        assert 'missing_data_handling' in result
        assert 'imputed_values' in result
        assert 'quality_metrics' in result
    
    def test_handle_extreme_values(self, orchestrator, data_generator):
        """Test handling of extreme values in data."""
        data = data_generator.generate_sales_data()
        # Add extreme values
        data.loc[20:22, 'sales'] = 1000000  # Very high values
        data.loc[25:27, 'sales'] = -1000  # Negative values
        
        result = orchestrator.process({
            'action': 'analyze_data',
            'data': data,
            'metrics': ['sales']
        })
        
        assert 'outlier_detection' in result
        assert 'data_quality_issues' in result
        assert 'recommendations' in result

if __name__ == "__main__":
    pytest.main([__file__]) 