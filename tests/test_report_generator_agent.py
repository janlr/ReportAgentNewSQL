import pytest
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import os
import numpy as np

from agents.report_generator_agent import ReportGeneratorAgent, ReportTemplate

@pytest.fixture
def sample_config():
    return {
        "reports_dir": "./test_reports",
        "client_configs": {
            "default": {
                "report_categories": ["sales", "inventory"],
                "metrics_mapping": {
                    "sales": {
                        "total_revenue": "SUM(OrderDetails.UnitPrice * OrderDetails.Quantity)",
                        "order_count": "COUNT(DISTINCT Orders.OrderID)"
                    }
                },
                "table_mapping": {
                    "orders": "Sales.Orders",
                    "order_details": "Sales.OrderDetails"
                },
                "field_mapping": {
                    "order_date": "OrderDate",
                    "product_name": "ProductName"
                }
            }
        }
    }

@pytest.fixture
def agent(sample_config):
    return ReportGeneratorAgent(sample_config)

@pytest.fixture
def report_generator():
    config = {
        "template_dir": "./templates",
        "output_dir": "./reports"
    }
    return ReportGeneratorAgent(
        config=config,
        output_dir="./reports",
        anthropic_api_key="test_key"
    )

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "OrderDate": pd.date_range(start="2024-01-01", periods=10),
        "LineTotal": np.random.uniform(100, 1000, 10),
        "CategoryName": ["Category A", "Category B"] * 5,
        "Territory": ["North", "South", "East", "West"] * 3
    })

class TestNaturalLanguageProcessing:
    """Test cases for natural language prompt parsing."""
    
    def test_parse_time_filters(self, agent):
        """Test parsing of time-related phrases."""
        prompt = "Show sales for last 30 days and compare with last 3 months"
        result = agent.parse_user_prompt(prompt)
        
        assert "time_filters" in result
        assert result["time_filters"]["last_n_days"] == "30"
        assert result["time_filters"]["last_n_months"] == "3"
    
    def test_parse_metrics(self, agent):
        """Test parsing of metric-related phrases."""
        prompt = "Show total sales revenue and unique customers with average order value"
        result = agent.parse_user_prompt(prompt)
        
        assert "metrics" in result
        assert "sales" in result["metrics"]
        assert "customers" in result["metrics"]
        assert "average" in result["metrics"]
    
    def test_parse_dimensions(self, agent):
        """Test parsing of dimension-related phrases."""
        prompt = "Show sales by product and region, grouped by category"
        result = agent.parse_user_prompt(prompt)
        
        assert "dimensions" in result
        assert "product" in result["dimensions"]
        assert "region" in result["dimensions"]
        assert "category" in result["dimensions"]
    
    def test_parse_sorting(self, agent):
        """Test parsing of sorting preferences."""
        prompt = "Show top 10 products sorted by revenue"
        result = agent.parse_user_prompt(prompt)
        
        assert "sorting" in result
        assert result["sorting"]["top"] == "10"
        assert "sort" in result["sorting"]

class TestQueryOptimization:
    """Test cases for query optimization features."""
    
    def test_index_hint_addition(self, agent):
        """Test addition of index hints for large tables."""
        query = "SELECT * FROM Sales.Orders"
        optimized, hints = agent.optimize_query(query)
        
        assert "WITH (INDEX(IX_OrderDate))" in optimized
        assert "Added index hint for OrderDate" in hints
    
    def test_recompile_option(self, agent):
        """Test addition of RECOMPILE option for complex queries."""
        query = "SELECT * FROM Orders o JOIN OrderDetails od ON o.ID = od.OrderID JOIN Products p ON od.ProductID = p.ID"
        optimized, hints = agent.optimize_query(query)
        
        assert "OPTION (RECOMPILE)" in optimized
        assert "Added RECOMPILE hint for complex query" in hints
    
    def test_result_limit(self, agent):
        """Test addition of TOP clause for large result sets."""
        query = "SELECT * FROM Orders ORDER BY OrderDate"
        optimized, hints = agent.optimize_query(query)
        
        assert "TOP 10000" in optimized
        assert "Added TOP clause to limit result set" in hints
    
    def test_parallel_execution(self, agent):
        """Test addition of parallel execution hints."""
        query = "SELECT * FROM Orders o JOIN OrderDetails od ON o.ID = od.OrderID"
        optimized, hints = agent.optimize_query(query)
        
        assert "OPTION (MAXDOP 4)" in optimized
        assert "Added parallel execution hint" in hints

class TestConfigurationManagement:
    """Test cases for client configuration management."""
    
    def test_validate_config_structure(self, agent):
        """Test validation of client configuration structure."""
        valid_config = {
            "report_categories": ["sales"],
            "metrics_mapping": {"sales": {}},
            "table_mapping": {},
            "field_mapping": {}
        }
        assert agent.validate_client_config(valid_config) is True
        
        invalid_config = {
            "report_categories": ["sales"]
            # Missing required keys
        }
        assert agent.validate_client_config(invalid_config) is False
    
    def test_example_config_creation(self, agent):
        """Test creation of example configurations."""
        agent.create_example_configs()
        
        assert "ecommerce_example" in agent.client_configs
        assert "manufacturing_example" in agent.client_configs
        
        ecommerce = agent.client_configs["ecommerce_example"]
        assert "sales" in ecommerce["report_categories"]
        assert "total_revenue" in ecommerce["metrics_mapping"]["sales"]
    
    def test_config_export_import(self, agent, tmp_path):
        """Test export and import of client configurations."""
        # Export config
        export_path = tmp_path / "test_config_export.json"
        success = agent.export_client_config("default", export_path)
        assert success is True
        assert export_path.exists()
        
        # Import config
        success = agent.import_client_config(export_path, "imported_client")
        assert success is True
        assert "imported_client" in agent.client_configs

class TestComplexReportGeneration:
    """Test cases for complex report generation scenarios."""
    
    def test_natural_language_report(self, agent):
        """Test report generation from natural language prompt."""
        prompt = "Show total sales revenue by product category for last 3 months, " \
                "include customer count and average order value, " \
                "sort by revenue descending, limit to top 10"
        
        result = agent.process({
            "action": "generate_report",
            "prompt": prompt
        })
        
        assert result["status"] == "success"
        assert "query" in result
        assert "optimizations" in result
        
        # Verify query components
        query = result["query"]
        assert "SUM(OrderDetails.UnitPrice * OrderDetails.Quantity)" in query
        assert "COUNT(DISTINCT CustomerID)" in query
        assert "GROUP BY" in query
        assert "ORDER BY" in query
        assert "TOP 10" in query
    
    def test_template_based_report(self, agent):
        """Test report generation from template with complex parameters."""
        template = ReportTemplate(
            id="test_template",
            name="Complex Sales Report",
            description="Test template",
            category="sales",
            tags=["test"],
            parameters={
                "metrics": ["total_revenue", "order_count"],
                "dimensions": ["category", "region"],
                "main_table": "orders"
            },
            joins=[{
                "type": "INNER",
                "table": "order_details",
                "from_field": "order_id",
                "join_field": "order_id"
            }],
            group_by=["category", "region"],
            having={"total_revenue": "> 1000"},
            order_by=[{"field": "total_revenue", "direction": "DESC"}]
        )
        
        result = agent.process({
            "action": "generate_report",
            "template": template
        })
        
        assert result["status"] == "success"
        assert "query" in result
        assert "optimizations" in result
        
        # Verify query structure
        query = result["query"]
        assert "INNER JOIN" in query
        assert "GROUP BY" in query
        assert "HAVING" in query
        assert "ORDER BY" in query
    
    def test_error_handling(self, agent):
        """Test error handling in report generation."""
        # Test with invalid prompt
        result = agent.process({
            "action": "generate_report",
            "prompt": ""  # Empty prompt
        })
        assert result["status"] == "error"
        
        # Test with invalid template
        result = agent.process({
            "action": "generate_report",
            "template_id": "non_existent"
        })
        assert result["status"] == "error"
        
        # Test with invalid configuration
        with pytest.raises(Exception):
            agent.generate_query(None)

@pytest.mark.asyncio
async def test_initialization(report_generator):
    success = await report_generator.initialize()
    assert success == True
    assert Path("./test_reports").exists()

@pytest.mark.asyncio
async def test_process_invalid_input(report_generator):
    result = await report_generator.process({"action": "invalid"})
    assert result["success"] == False
    assert "error" in result

@pytest.mark.asyncio
async def test_cleanup(report_generator):
    success = await report_generator.cleanup()
    assert success == True

@patch('anthropic.Client')
@pytest.mark.asyncio
async def test_generate_insights(mock_anthropic, report_generator, sample_data):
    # Setup mock response
    mock_client = MagicMock()
    mock_client.generate_content.return_value.text = "Test insights"
    mock_anthropic.return_value = mock_client
    
    # Test insights generation
    parameters = {"report_type": "Sales Analysis"}
    data_tables = {"sales_data": sample_data}
    
    insights = await report_generator._generate_insights(data_tables, parameters)
    
    assert insights == "Test insights"
    mock_client.generate_content.assert_called_once()

@patch('anthropic.Client')
@pytest.mark.asyncio
async def test_generate_summary(mock_anthropic, report_generator, sample_data):
    # Setup mock response
    mock_client = MagicMock()
    mock_client.generate_content.return_value.text = "Test summary"
    mock_anthropic.return_value = mock_client
    
    # Test summary generation
    parameters = {"report_type": "Sales Analysis"}
    data_tables = {"sales_data": sample_data}
    
    summary = await report_generator._generate_summary(data_tables, parameters)
    
    assert summary == "Test summary"
    mock_client.generate_content.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__]) 