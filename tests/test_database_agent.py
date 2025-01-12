import pytest
from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from agents.database_agent import DatabaseAgent

@pytest.fixture
def sample_config():
    return {
        "db_config": {
            "driver": "ODBC Driver 17 for SQL Server",
            "host": "localhost",
            "port": 1433,
            "database": "TestDB",
            "username": "test_user",
            "password": "test_pass",
            "encrypt": "yes",
            "trust_server_certificate": "yes"
        },
        "cache_dir": "./test_cache"
    }

@pytest.fixture
def agent(sample_config):
    return DatabaseAgent(sample_config)

class TestDatabaseConnection:
    """Test cases for database connection management."""
    
    def test_connection_initialization(self, agent):
        """Test database connection initialization."""
        assert agent.db_config is not None
        assert agent.cache_dir == Path("./test_cache")
    
    @patch('sqlalchemy.create_engine')
    def test_successful_connection(self, mock_create_engine, agent):
        """Test successful database connection."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        success = agent.initialize()
        assert success is True
        assert agent.engine is not None
        mock_create_engine.assert_called_once()
    
    @patch('sqlalchemy.create_engine')
    def test_connection_error(self, mock_create_engine, agent):
        """Test database connection error handling."""
        mock_create_engine.side_effect = SQLAlchemyError("Connection failed")
        
        success = agent.initialize()
        assert success is False
        assert agent.engine is None

class TestSchemaDiscovery:
    """Test cases for schema discovery functionality."""
    
    @patch('sqlalchemy.inspect')
    def test_get_schema_info(self, mock_inspect, agent):
        """Test retrieval of schema information."""
        mock_inspector = Mock()
        mock_inspect.return_value = mock_inspector
        
        # Mock table list
        mock_inspector.get_table_names.return_value = ["Orders", "Products"]
        # Mock column information
        mock_inspector.get_columns.return_value = [
            {"name": "OrderID", "type": "INTEGER", "primary_key": True},
            {"name": "OrderDate", "type": "DATETIME"}
        ]
        
        schema_info = agent.get_schema_info()
        assert "tables" in schema_info
        assert len(schema_info["tables"]) == 2
        assert "Orders" in schema_info["tables"]
        assert "columns" in schema_info["tables"]["Orders"]
    
    @patch('sqlalchemy.inspect')
    def test_get_relationships(self, mock_inspect, agent):
        """Test discovery of table relationships."""
        mock_inspector = Mock()
        mock_inspect.return_value = mock_inspector
        
        # Mock foreign key information
        mock_inspector.get_foreign_keys.return_value = [{
            "referred_table": "Products",
            "referred_columns": ["ProductID"],
            "constrained_columns": ["ProductID"]
        }]
        
        relationships = agent.get_relationships("Orders")
        assert len(relationships) > 0
        assert "Products" in str(relationships)

class TestQueryExecution:
    """Test cases for query execution functionality."""
    
    @patch('pandas.read_sql')
    def test_execute_query(self, mock_read_sql, agent):
        """Test execution of SQL queries."""
        mock_df = pd.DataFrame({
            "OrderID": [1, 2],
            "Total": [100.0, 200.0]
        })
        mock_read_sql.return_value = mock_df
        
        query = "SELECT OrderID, Total FROM Orders"
        result = agent.execute_query(query)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "OrderID" in result.columns
        mock_read_sql.assert_called_once()
    
    @patch('pandas.read_sql')
    def test_query_error(self, mock_read_sql, agent):
        """Test error handling in query execution."""
        mock_read_sql.side_effect = SQLAlchemyError("Query failed")
        
        query = "SELECT * FROM NonExistentTable"
        with pytest.raises(Exception):
            agent.execute_query(query)

class TestDataRetrieval:
    """Test cases for specific data retrieval methods."""
    
    @patch('pandas.read_sql')
    def test_get_table_sample(self, mock_read_sql, agent):
        """Test retrieval of table samples."""
        mock_df = pd.DataFrame({"ID": range(5)})
        mock_read_sql.return_value = mock_df
        
        sample = agent.get_table_sample("Orders", 5)
        assert len(sample) == 5
        mock_read_sql.assert_called_once()
    
    @patch('pandas.read_sql')
    def test_get_table_stats(self, mock_read_sql, agent):
        """Test retrieval of table statistics."""
        mock_df = pd.DataFrame({
            "column_name": ["OrderID"],
            "row_count": [1000],
            "distinct_count": [1000]
        })
        mock_read_sql.return_value = mock_df
        
        stats = agent.get_table_stats("Orders")
        assert "row_count" in stats
        assert "distinct_counts" in stats

class TestCacheManagement:
    """Test cases for query result caching."""
    
    def test_cache_query_result(self, agent):
        """Test caching of query results."""
        query = "SELECT * FROM Orders"
        df = pd.DataFrame({"ID": range(5)})
        
        # Cache the result
        agent.cache_query_result(query, df)
        
        # Verify cache exists
        assert agent.is_query_cached(query)
        
        # Retrieve from cache
        cached_df = agent.get_cached_result(query)
        assert cached_df is not None
        assert len(cached_df) == len(df)
    
    def test_cache_invalidation(self, agent):
        """Test cache invalidation."""
        query = "SELECT * FROM Orders"
        df = pd.DataFrame({"ID": range(5)})
        
        # Cache and then invalidate
        agent.cache_query_result(query, df)
        agent.invalidate_cache(query)
        
        assert not agent.is_query_cached(query)

if __name__ == "__main__":
    pytest.main([__file__]) 