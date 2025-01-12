import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from database import DatabaseConnection
from sqlalchemy import create_engine
import sqlite3
import tempfile
import os

@pytest.fixture
def sample_configs():
    return {
        "mssql": {
            "driver": "ODBC Driver 17 for SQL Server",
            "server": "localhost",
            "database": "TestDB",
            "trusted_connection": "yes"
        },
        "mysql": {
            "driver": "MySQL ODBC 8.0 Driver",
            "server": "localhost",
            "database": "test_db",
            "uid": "test_user",
            "pwd": "test_pass"
        },
        "postgres": {
            "driver": "PostgreSQL UNICODE",
            "server": "localhost",
            "database": "test_db",
            "uid": "test_user",
            "pwd": "test_pass"
        },
        "sqlite": {
            "driver": "SQLite3",
            "database": ":memory:"
        }
    }

@pytest.fixture
def sample_data():
    return {
        "sales": pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=10),
            "product": [f"Product_{i}" for i in range(10)],
            "quantity": np.random.randint(1, 100, 10),
            "price": np.random.uniform(10, 1000, 10)
        }),
        "customers": pd.DataFrame({
            "id": range(1, 11),
            "name": [f"Customer_{i}" for i in range(1, 11)],
            "country": np.random.choice(["USA", "UK", "Canada"], 10)
        })
    }

@pytest.fixture
def sqlite_db():
    """Create a temporary SQLite database for testing."""
    db_fd, db_path = tempfile.mkstemp()
    conn = sqlite3.connect(db_path)
    
    # Create test tables
    conn.execute("""
        CREATE TABLE sales (
            date TEXT,
            product TEXT,
            quantity INTEGER,
            price REAL
        )
    """)
    
    conn.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT
        )
    """)
    
    conn.close()
    
    yield db_path
    
    os.close(db_fd)
    os.unlink(db_path)

class TestDatabaseConnection:
    def test_supported_drivers(self):
        """Test that supported drivers are correctly defined."""
        assert "MSSQL" in DatabaseConnection.SUPPORTED_DRIVERS
        assert "MySQL" in DatabaseConnection.SUPPORTED_DRIVERS
        assert "PostgreSQL" in DatabaseConnection.SUPPORTED_DRIVERS
        assert "SQLite" in DatabaseConnection.SUPPORTED_DRIVERS
    
    def test_init(self):
        """Test database connection initialization."""
        config = {"driver": "SQLite3", "database": ":memory:"}
        db = DatabaseConnection(config)
        assert db.config == config
        assert db.connection is None
        assert db.engine is None
    
    def test_sqlite_connection(self, sqlite_db):
        """Test SQLite database connection."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        assert db.connection is not None
        assert db.engine is not None
        assert db.inspector is not None
        
        db.close()
    
    @patch('pyodbc.connect')
    def test_mssql_connection(self, mock_connect, sample_configs):
        """Test MSSQL database connection."""
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        db = DatabaseConnection(sample_configs["mssql"])
        db.connect()
        
        mock_connect.assert_called_once()
        assert "DRIVER" in mock_connect.call_args[0][0]
        assert "SERVER" in mock_connect.call_args[0][0]
        
        db.close()
    
    def test_schema_info(self, sqlite_db):
        """Test schema information retrieval."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        schema_info = db.get_schema_info()
        
        assert "tables" in schema_info
        assert "views" in schema_info
        assert "relationships" in schema_info
        assert len(schema_info["tables"]) >= 2  # sales and customers tables
        
        # Check table details
        sales_table = next(t for t in schema_info["tables"] if t["name"] == "sales")
        assert len(sales_table["columns"]) == 4
        assert any(c["name"] == "product" for c in sales_table["columns"])
        
        db.close()
    
    def test_execute_query(self, sqlite_db, sample_data):
        """Test query execution and DataFrame retrieval."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        # Insert sample data
        sample_data["sales"].to_sql("sales", db.engine, if_exists="replace", index=False)
        
        # Test simple query
        query = "SELECT * FROM sales WHERE quantity > :min_quantity"
        params = {"min_quantity": 50}
        
        result = db.execute_query(query, params)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data["sales"][sample_data["sales"]["quantity"] > 50])
        assert all(result["quantity"] > 50)
        
        db.close()
    
    def test_connection_error(self):
        """Test handling of connection errors."""
        config = {
            "driver": "SQLite3",
            "database": "nonexistent.db"
        }
        db = DatabaseConnection(config)
        
        with pytest.raises(Exception):
            db.connect()
    
    def test_query_error(self, sqlite_db):
        """Test handling of query errors."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        with pytest.raises(Exception):
            db.execute_query("SELECT * FROM nonexistent_table")
        
        db.close()
    
    def test_get_relationships(self, sqlite_db):
        """Test relationship detection."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        # Create tables with relationships
        db.execute_query("""
            CREATE TABLE categories (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        
        db.execute_query("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                category_id INTEGER,
                name TEXT,
                FOREIGN KEY (category_id) REFERENCES categories (id)
            )
        """)
        
        relationships = db.get_schema_info()["relationships"]
        
        assert len(relationships) > 0
        relationship = relationships[0]
        assert relationship["source_table"] == "products"
        assert relationship["target_table"] == "categories"
        
        db.close()
    
    def test_connection_cleanup(self, sqlite_db):
        """Test proper cleanup of database connections."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        assert db.connection is not None
        assert db.engine is not None
        
        db.close()
        
        assert db.connection is None
        assert db.engine is None
    
    def test_query_optimization(self, sqlite_db):
        """Test query optimization features."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        # Create test data
        db.execute_query("""
            CREATE TABLE test_data (
                id INTEGER PRIMARY KEY,
                value INTEGER,
                category TEXT
            )
        """)
        
        # Insert test data
        values = [(i, i % 100, f"Category_{i % 5}") for i in range(1000)]
        db.execute_query(
            "INSERT INTO test_data (id, value, category) VALUES (?, ?, ?)",
            params=values
        )
        
        # Test query with optimization
        query = """
            SELECT category, AVG(value) as avg_value
            FROM test_data
            GROUP BY category
            HAVING avg_value > :min_value
        """
        
        result = db.execute_query(query, {"min_value": 50}, optimize=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(result["avg_value"] > 50)
        
        db.close()
    
    def test_performance_monitoring(self, sqlite_db, sample_data):
        """Test query performance monitoring."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        # Execute multiple queries
        sample_data["sales"].to_sql("sales", db.engine, if_exists="replace", index=False)
        
        queries = [
            "SELECT * FROM sales",
            "SELECT COUNT(*) FROM sales",
            "SELECT AVG(price) FROM sales",
            "SELECT product, SUM(quantity) FROM sales GROUP BY product"
        ]
        
        for query in queries:
            db.execute_query(query)
        
        # Get performance analysis
        analysis = db.analyze_query_performance()
        
        assert "total_queries" in analysis
        assert analysis["total_queries"] == len(queries)
        assert "avg_execution_time" in analysis
        assert "performance_summary" in analysis
        assert len(analysis["performance_summary"]) > 0
        
        db.close()
    
    def test_query_optimization_mssql(self, sample_configs):
        """Test MSSQL specific query optimizations."""
        with patch('pyodbc.connect') as mock_connect:
            mock_connection = Mock()
            mock_connect.return_value = mock_connection
            
            db = DatabaseConnection(sample_configs["mssql"])
            db.connect()
            
            # Test aggregation query optimization
            query = "SELECT department, SUM(salary) FROM employees GROUP BY department"
            optimized = db._optimize_mssql_query(query, Mock(get_type=lambda: 'SELECT'))
            
            assert "OPTION (HASH GROUP)" in optimized
            
            # Test join query optimization
            query = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id"
            optimized = db._optimize_mssql_query(query, Mock(get_type=lambda: 'SELECT'))
            
            assert "OPTION (HASH JOIN)" in optimized
            
            db.close()
    
    def test_query_optimization_mysql(self, sample_configs):
        """Test MySQL specific query optimizations."""
        with patch('pymysql.connect') as mock_connect:
            mock_connection = Mock()
            mock_connect.return_value = mock_connection
            
            db = DatabaseConnection(sample_configs["mysql"])
            db.connect()
            
            # Test complex join optimization
            query = """
                SELECT * FROM orders o 
                JOIN customers c ON o.customer_id = c.id
                JOIN products p ON o.product_id = p.id
                JOIN categories cat ON p.category_id = cat.id
            """
            optimized = db._optimize_mysql_query(query, Mock(get_type=lambda: 'SELECT'))
            
            assert "STRAIGHT_JOIN" in optimized
            
            db.close()
    
    def test_query_optimization_postgres(self, sample_configs):
        """Test PostgreSQL specific query optimizations."""
        with patch('psycopg2.connect') as mock_connect:
            mock_connection = Mock()
            mock_connect.return_value = mock_connection
            
            db = DatabaseConnection(sample_configs["postgres"])
            db.connect()
            
            # Test parallel query optimization
            query = "SELECT * FROM large_table"
            parsed = Mock(get_type=lambda: 'SELECT')
            with patch.object(db, '_is_large_table_query', return_value=True):
                with patch.object(db, '_can_parallelize', return_value=True):
                    optimized = db._optimize_postgres_query(query, parsed)
                    
                    assert "max_parallel_workers_per_gather" in optimized
            
            db.close()
    
    def test_performance_metrics(self, sqlite_db):
        """Test query performance metrics collection."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        # Execute a query and check metrics
        query = "SELECT 1"
        db.execute_query(query)
        
        metrics = db._query_metrics[-1]
        assert "query_hash" in metrics
        assert "execution_time" in metrics
        assert "row_count" in metrics
        assert "timestamp" in metrics
        
        # Test metrics analysis
        analysis = db.analyze_query_performance()
        assert analysis["total_queries"] == 1
        assert "avg_execution_time" in analysis
        assert "performance_summary" in analysis
        
        db.close()
    
    def test_error_handling_in_optimization(self, sqlite_db):
        """Test error handling in query optimization."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        # Test with invalid SQL
        invalid_query = "INVALID SQL QUERY"
        result = db._optimize_query(invalid_query)
        
        # Should return original query if optimization fails
        assert result == invalid_query
        
        db.close()
    
    def test_query_caching_detection(self, sqlite_db):
        """Test detection of cacheable queries."""
        config = {
            "driver": "SQLite3",
            "database": sqlite_db
        }
        db = DatabaseConnection(config)
        db.connect()
        
        # Test static query
        static_query = "SELECT * FROM static_table"
        parsed = Mock(get_type=lambda: 'SELECT')
        assert db._is_cacheable_query(parsed)
        
        # Test non-cacheable query
        update_query = "UPDATE table SET column = value"
        parsed = Mock(get_type=lambda: 'UPDATE')
        assert not db._is_cacheable_query(parsed)
        
        db.close()

if __name__ == "__main__":
    pytest.main([__file__]) 