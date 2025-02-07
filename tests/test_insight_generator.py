import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from unittest.mock import Mock, patch, AsyncMock
from agents.insight_generator_agent import InsightGeneratorAgent

@pytest.fixture
def mock_llm_manager():
    """Create a mock LLM manager."""
    mock_llm = AsyncMock()
    mock_llm.active_provider = "test_provider"
    mock_llm.initialize.return_value = True
    mock_llm.cleanup.return_value = True
    mock_llm.generate_async.return_value = """
    Overview:
    The data shows strong performance across key metrics.
    
    Key Findings:
    - Sales increased by 25% year-over-year
    - Customer retention rate is at 85%
    - Top product category is Electronics
    
    Recommendations:
    - Expand the Electronics category
    - Focus on customer retention programs
    - Optimize inventory levels
    """
    return mock_llm

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create date range
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    
    # Generate sample data
    np.random.seed(42)
    data = {
        "date": dates,
        "sales": np.random.normal(1000, 200, len(dates)),
        "customers": np.random.normal(100, 20, len(dates)),
        "category": np.random.choice(["Electronics", "Clothing", "Books"], len(dates)),
        "region": np.random.choice(["North", "South", "East", "West"], len(dates))
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def insight_agent(tmp_path, mock_llm_manager):
    """Create an instance of InsightGeneratorAgent for testing."""
    config = {
        "llm_manager": {
            "provider": "test_provider",
            "model": "test_model",
            "api_key": "test_key",
            "cache_dir": str(tmp_path / "llm_cache")
        },
        "cache_dir": str(tmp_path / "insights")
    }
    
    with patch("agents.insight_generator_agent.LLMManagerAgent", return_value=mock_llm_manager):
        agent = InsightGeneratorAgent(config)
    return agent

@pytest.mark.asyncio
async def test_initialization(insight_agent):
    """Test agent initialization."""
    result = await insight_agent.initialize()
    assert result is True
    assert insight_agent.cache_dir.exists()

@pytest.mark.asyncio
async def test_generate_summary(insight_agent, sample_data):
    """Test summary generation."""
    input_data = {
        "action": "generate_summary",
        "data": sample_data.to_dict(orient="records"),
        "metadata": {
            "report_type": "sales",
            "query_context": "SELECT * FROM sales",
            "rows_processed": len(sample_data)
        }
    }
    
    result = await insight_agent.process(input_data)
    
    assert result["success"] is True
    assert "summary" in result
    assert "overview" in result["summary"]
    assert "key_findings" in result["summary"]
    assert "recommendations" in result["summary"]
    assert len(result["summary"]["key_findings"]) > 0
    assert len(result["summary"]["recommendations"]) > 0

@pytest.mark.asyncio
async def test_summary_caching(insight_agent, sample_data):
    """Test that summaries are properly cached."""
    input_data = {
        "action": "generate_summary",
        "data": sample_data.to_dict(orient="records"),
        "metadata": {
            "report_type": "sales",
            "query_context": "SELECT * FROM sales"
        }
    }
    
    # First call should not be from cache
    result1 = await insight_agent.process(input_data)
    assert result1["success"] is True
    assert result1.get("from_cache") is False
    
    # Second call should be from cache
    result2 = await insight_agent.process(input_data)
    assert result2["success"] is True
    assert result2.get("from_cache") is True
    
    # Results should be identical
    assert result1["summary"] == result2["summary"]

@pytest.mark.asyncio
async def test_different_report_types(insight_agent, sample_data):
    """Test summary generation for different report types."""
    report_types = ["sales", "inventory", "general"]
    
    for report_type in report_types:
        input_data = {
            "action": "generate_summary",
            "data": sample_data.to_dict(orient="records"),
            "metadata": {
                "report_type": report_type,
                "query_context": f"SELECT * FROM {report_type}"
            }
        }
        
        result = await insight_agent.process(input_data)
        assert result["success"] is True
        assert "summary" in result

@pytest.mark.asyncio
async def test_invalid_input(insight_agent):
    """Test handling of invalid input."""
    invalid_inputs = [
        {"action": "generate_summary"},  # Missing data and metadata
        {"action": "unknown_action"},    # Invalid action
        {"action": "generate_summary", "data": []},  # Missing metadata
    ]
    
    for input_data in invalid_inputs:
        result = await insight_agent.process(input_data)
        assert result["success"] is False
        assert "error" in result

@pytest.mark.asyncio
async def test_statistics_calculation(insight_agent, sample_data):
    """Test calculation of statistics from data."""
    stats = insight_agent._calculate_statistics(sample_data)
    
    assert "row_count" in stats
    assert "column_count" in stats
    assert "numeric_columns" in stats
    assert "categorical_columns" in stats
    
    assert stats["row_count"] == len(sample_data)
    assert stats["column_count"] == len(sample_data.columns)
    assert "sales" in stats["numeric_columns"]
    assert "category" in stats["categorical_columns"]

@pytest.mark.asyncio
async def test_prompt_generation(insight_agent):
    """Test generation of prompts for different report types."""
    stats = {
        "row_count": 100,
        "numeric_columns": {"sales": {"mean": 1000}},
        "categorical_columns": {"category": {"unique_values": 3}}
    }
    metadata = {"query_context": "SELECT * FROM sales"}
    
    # Test sales prompt
    sales_prompt = insight_agent._generate_prompt("sales", stats, metadata)
    assert "sales performance" in sales_prompt.lower()
    
    # Test inventory prompt
    inventory_prompt = insight_agent._generate_prompt("inventory", stats, metadata)
    assert "inventory status" in inventory_prompt.lower()
    
    # Test general prompt
    general_prompt = insight_agent._generate_prompt("general", stats, metadata)
    assert "patterns and trends" in general_prompt.lower()

@pytest.mark.asyncio
async def test_extract_findings_and_recommendations(insight_agent):
    """Test extraction of findings and recommendations from LLM response."""
    content = """
    Overview:
    Performance summary.
    
    Key Findings:
    - Finding 1
    - Finding 2
    * Finding 3
    
    Recommendations:
    1. Recommendation 1
    2. Recommendation 2
    - Recommendation 3
    """
    
    findings = insight_agent._extract_key_findings(content)
    recommendations = insight_agent._extract_recommendations(content)
    
    assert len(findings) == 3
    assert len(recommendations) == 3
    assert "Finding 1" in findings
    assert "Recommendation 1" in recommendations

@pytest.mark.asyncio
async def test_time_column_detection(insight_agent, sample_data):
    """Test detection of time-based columns."""
    # Test with time column
    assert insight_agent._has_time_column(sample_data) is True
    
    # Test without time column
    data_no_time = sample_data.drop("date", axis=1)
    assert insight_agent._has_time_column(data_no_time) is False 