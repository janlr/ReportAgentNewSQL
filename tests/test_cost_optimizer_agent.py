import pytest
from pathlib import Path
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from agents.cost_optimizer_agent import CostOptimizerAgent

@pytest.fixture
def sample_config():
    return {
        "stats_dir": "./test_stats",
        "cost_limits": {
            "daily": 10.0,
            "monthly": 100.0
        },
        "cache_config": {
            "max_size": 1000,
            "ttl": 3600
        }
    }

@pytest.fixture
def agent(sample_config):
    return CostOptimizerAgent(sample_config)

class TestCostTracking:
    """Test cases for cost tracking functionality."""
    
    def test_track_api_usage(self, agent):
        """Test tracking of API usage."""
        usage_data = {
            "model": "gpt-4",
            "tokens": 1000,
            "cost": 0.03,
            "duration": 2.5
        }
        
        tracked = agent.track_api_usage(usage_data)
        assert tracked is True
        
        stats = agent.get_usage_stats("api")
        assert stats["total_tokens"] >= 1000
        assert stats["total_cost"] >= 0.03
    
    def test_track_query_execution(self, agent):
        """Test tracking of database query execution."""
        query_data = {
            "query": "SELECT * FROM Orders",
            "duration": 1.5,
            "rows_returned": 1000,
            "cache_hit": False
        }
        
        tracked = agent.track_query_execution(query_data)
        assert tracked is True
        
        stats = agent.get_usage_stats("query")
        assert stats["total_queries"] >= 1
        assert stats["total_rows"] >= 1000
    
    def test_track_report_generation(self, agent):
        """Test tracking of report generation."""
        report_data = {
            "report_type": "sales",
            "components": 5,
            "duration": 3.0,
            "size": 1024
        }
        
        tracked = agent.track_report_generation(report_data)
        assert tracked is True
        
        stats = agent.get_usage_stats("report")
        assert stats["total_reports"] >= 1
        assert stats["total_components"] >= 5

class TestCostLimits:
    """Test cases for cost limit management."""
    
    def test_daily_limit_check(self, agent):
        """Test checking against daily cost limits."""
        # Add usage close to limit
        for _ in range(9):
            agent.track_api_usage({
                "model": "gpt-4",
                "tokens": 1000,
                "cost": 1.0,
                "duration": 1.0
            })
        
        # Check if next usage would exceed limit
        would_exceed = agent.would_exceed_limit({
            "model": "gpt-4",
            "tokens": 1000,
            "cost": 2.0
        })
        assert would_exceed is True
    
    def test_monthly_limit_check(self, agent):
        """Test checking against monthly cost limits."""
        # Add usage close to limit
        for _ in range(90):
            agent.track_api_usage({
                "model": "gpt-4",
                "tokens": 1000,
                "cost": 1.0,
                "duration": 1.0
            })
        
        # Check if next usage would exceed limit
        would_exceed = agent.would_exceed_limit({
            "model": "gpt-4",
            "tokens": 1000,
            "cost": 20.0
        }, period="monthly")
        assert would_exceed is True

class TestOptimizationRecommendations:
    """Test cases for optimization recommendations."""
    
    def test_get_api_recommendations(self, agent):
        """Test generation of API usage optimization recommendations."""
        # Add some usage data
        for _ in range(10):
            agent.track_api_usage({
                "model": "gpt-4",
                "tokens": 1000,
                "cost": 0.03,
                "duration": 2.5
            })
        
        recommendations = agent.get_optimization_recommendations("api")
        assert len(recommendations) > 0
        assert any("cost" in r.lower() for r in recommendations)
    
    def test_get_query_recommendations(self, agent):
        """Test generation of query optimization recommendations."""
        # Add some query execution data
        for _ in range(10):
            agent.track_query_execution({
                "query": "SELECT * FROM Orders",
                "duration": 1.5,
                "rows_returned": 1000,
                "cache_hit": False
            })
        
        recommendations = agent.get_optimization_recommendations("query")
        assert len(recommendations) > 0
        assert any("cache" in r.lower() for r in recommendations)

class TestCacheOptimization:
    """Test cases for cache optimization."""
    
    def test_optimize_cache_usage(self, agent):
        """Test optimization of cache usage."""
        # Add some items to cache
        for i in range(10):
            agent.cache_data(f"key_{i}", f"value_{i}")
        
        # Optimize cache
        removed = agent.optimize_cache()
        assert isinstance(removed, int)
        assert removed >= 0
    
    def test_cache_eviction(self, agent):
        """Test cache eviction based on TTL."""
        # Add item with expired TTL
        with patch('time.time') as mock_time:
            mock_time.return_value = datetime.now().timestamp()
            agent.cache_data("test_key", "test_value")
            
            # Move time forward past TTL
            mock_time.return_value += agent.config["cache_config"]["ttl"] + 100
            
            # Check if item is evicted
            assert agent.get_cached_data("test_key") is None

class TestUsageAnalytics:
    """Test cases for usage analytics functionality."""
    
    def test_get_usage_summary(self, agent):
        """Test retrieval of usage summary."""
        # Add some usage data
        agent.track_api_usage({
            "model": "gpt-4",
            "tokens": 1000,
            "cost": 0.03,
            "duration": 2.5
        })
        
        summary = agent.get_usage_summary()
        assert "api" in summary
        assert "total_cost" in summary["api"]
        assert "total_tokens" in summary["api"]
    
    def test_get_usage_trends(self, agent):
        """Test analysis of usage trends."""
        # Add usage data over multiple days
        dates = [datetime.now() - timedelta(days=i) for i in range(5)]
        for date in dates:
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.now.return_value = date
                agent.track_api_usage({
                    "model": "gpt-4",
                    "tokens": 1000,
                    "cost": 0.03,
                    "duration": 2.5
                })
        
        trends = agent.get_usage_trends()
        assert len(trends["dates"]) == 5
        assert len(trends["costs"]) == 5

class TestDataPersistence:
    """Test cases for usage data persistence."""
    
    @patch('json.dump')
    def test_save_usage_stats(self, mock_json_dump, agent):
        """Test saving of usage statistics."""
        agent.track_api_usage({
            "model": "gpt-4",
            "tokens": 1000,
            "cost": 0.03,
            "duration": 2.5
        })
        
        success = agent.save_usage_stats()
        assert success is True
        mock_json_dump.assert_called_once()
    
    @patch('pathlib.Path.exists')
    @patch('json.load')
    def test_load_usage_stats(self, mock_json_load, mock_exists, agent):
        """Test loading of usage statistics."""
        mock_exists.return_value = True
        mock_stats = {
            "api": {"total_cost": 1.0, "total_tokens": 1000},
            "query": {"total_queries": 10, "total_rows": 1000}
        }
        mock_json_load.return_value = mock_stats
        
        stats = agent.load_usage_stats()
        assert stats == mock_stats

if __name__ == "__main__":
    pytest.main([__file__]) 