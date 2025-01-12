import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from agents.cost_optimizer import CostOptimizerAgent

@pytest.fixture
def cost_optimizer():
    """Create a CostOptimizerAgent instance for testing."""
    return CostOptimizerAgent()

def test_track_api_usage(cost_optimizer):
    """Test tracking API usage statistics."""
    cost_optimizer.track_api_usage(
        api_name="openai",
        tokens_used=100,
        duration_ms=500,
        cost=0.01
    )
    
    assert len(cost_optimizer.usage_stats["api_calls"]) == 1
    call = cost_optimizer.usage_stats["api_calls"][0]
    assert call["api_name"] == "openai"
    assert call["tokens_used"] == 100
    assert call["duration_ms"] == 500
    assert call["cost"] == 0.01

def test_track_query_execution(cost_optimizer):
    """Test tracking query execution statistics."""
    cost_optimizer.track_query_execution(
        query="SELECT * FROM table",
        duration_ms=200,
        rows_processed=1000
    )
    
    assert len(cost_optimizer.usage_stats["query_executions"]) == 1
    query = cost_optimizer.usage_stats["query_executions"][0]
    assert query["query"] == "SELECT * FROM table"
    assert query["duration_ms"] == 200
    assert query["rows_processed"] == 1000

def test_track_report_generation(cost_optimizer):
    """Test tracking report generation statistics."""
    cost_optimizer.track_report_generation(
        report_type="sales_report",
        size_bytes=50000,
        duration_ms=1000,
        resources_used={"cpu": 0.5, "memory": 100}
    )
    
    assert len(cost_optimizer.usage_stats["report_generations"]) == 1
    report = cost_optimizer.usage_stats["report_generations"][0]
    assert report["report_type"] == "sales_report"
    assert report["size_bytes"] == 50000
    assert report["duration_ms"] == 1000
    assert report["resources_used"] == {"cpu": 0.5, "memory": 100}

def test_get_usage_summary(cost_optimizer):
    """Test getting usage statistics summary."""
    # Add some test data
    cost_optimizer.track_api_usage("openai", 100, 500, 0.01)
    cost_optimizer.track_api_usage("openai", 200, 600, 0.02)
    cost_optimizer.track_query_execution("SELECT 1", 100, 500)
    cost_optimizer.track_report_generation("test", 1000, 300, {})
    
    # Get summary
    summary = cost_optimizer.get_usage_summary()
    
    # Verify API usage summary
    assert summary["api_usage"]["total_calls"] == 2
    assert summary["api_usage"]["total_tokens"] == 300
    assert summary["api_usage"]["total_cost"] == 0.03
    assert summary["api_usage"]["avg_duration_ms"] == 550
    
    # Verify query usage summary
    assert summary["query_usage"]["total_queries"] == 1
    assert summary["query_usage"]["total_rows"] == 500
    assert summary["query_usage"]["avg_duration_ms"] == 100
    
    # Verify report usage summary
    assert summary["report_usage"]["total_reports"] == 1
    assert summary["report_usage"]["total_size_bytes"] == 1000
    assert summary["report_usage"]["avg_duration_ms"] == 300

def test_get_usage_summary_with_time_range(cost_optimizer):
    """Test getting usage summary with time range filtering."""
    # Add test data at different times
    now = datetime.now()
    
    # Manually set timestamps for testing
    cost_optimizer.usage_stats["api_calls"] = [
        {
            "timestamp": (now - timedelta(days=2)).isoformat(),
            "api_name": "openai",
            "tokens_used": 100,
            "duration_ms": 500,
            "cost": 0.01
        },
        {
            "timestamp": now.isoformat(),
            "api_name": "openai",
            "tokens_used": 200,
            "duration_ms": 600,
            "cost": 0.02
        }
    ]
    
    # Get summary for last day
    summary = cost_optimizer.get_usage_summary(
        start_time=now - timedelta(days=1)
    )
    
    assert summary["api_usage"]["total_calls"] == 1
    assert summary["api_usage"]["total_tokens"] == 200
    assert summary["api_usage"]["total_cost"] == 0.02

def test_get_optimization_recommendations(cost_optimizer):
    """Test getting optimization recommendations."""
    # Add test data that should trigger recommendations
    
    # High token usage
    for _ in range(5):
        cost_optimizer.track_api_usage("openai", 2000, 500, 0.04)
    
    # Slow queries
    for _ in range(3):
        cost_optimizer.track_query_execution("SELECT *", 2000, 1000)
    
    # Large reports
    for _ in range(2):
        cost_optimizer.track_report_generation(
            "test",
            20 * 1024 * 1024,  # 20MB
            1000,
            {}
        )
    
    recommendations = cost_optimizer.get_optimization_recommendations()
    
    # Verify we get recommendations
    assert len(recommendations) > 0
    
    # Check specific recommendation types
    rec_types = {r["type"] for r in recommendations}
    assert "api_optimization" in rec_types
    assert "query_optimization" in rec_types
    assert "report_optimization" in rec_types

def test_optimize_cache_usage(cost_optimizer):
    """Test cache optimization."""
    # Add some items to cache
    cost_optimizer.cache = {
        "key1": {"data": "x" * 1024 * 1024, "last_accessed": 100},  # 1MB
        "key2": {"data": "y" * 1024 * 1024, "last_accessed": 200},  # 1MB
        "key3": {"data": "z" * 1024 * 1024, "last_accessed": 300}   # 1MB
    }
    
    # Optimize cache with 2MB limit
    cost_optimizer.optimize_cache_usage(max_cache_size_mb=2)
    
    # Verify oldest item was removed
    assert "key1" not in cost_optimizer.cache
    assert "key2" in cost_optimizer.cache
    assert "key3" in cost_optimizer.cache

def test_save_and_load_usage_stats(cost_optimizer):
    """Test saving and loading usage statistics."""
    # Add some test data
    cost_optimizer.track_api_usage("openai", 100, 500, 0.01)
    cost_optimizer.track_query_execution("SELECT 1", 100, 500)
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        # Save stats
        cost_optimizer.save_usage_stats(temp_file.name)
        
        # Create new instance and load stats
        new_optimizer = CostOptimizerAgent()
        new_optimizer.load_usage_stats(temp_file.name)
        
        # Verify stats were loaded correctly
        assert len(new_optimizer.usage_stats["api_calls"]) == 1
        assert len(new_optimizer.usage_stats["query_executions"]) == 1
        
        # Clean up
        Path(temp_file.name).unlink()

def test_error_handling(cost_optimizer):
    """Test error handling in various scenarios."""
    # Invalid file path for loading
    with pytest.raises(Exception):
        cost_optimizer.load_usage_stats("nonexistent.json")
    
    # Invalid time range
    future_time = datetime.now() + timedelta(days=365)
    summary = cost_optimizer.get_usage_summary(start_time=future_time)
    assert summary["api_usage"]["total_calls"] == 0
    
    # Invalid cache size
    cost_optimizer.optimize_cache_usage(max_cache_size_mb=-1)
    assert len(cost_optimizer.cache) == 0 