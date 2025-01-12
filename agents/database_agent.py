from typing import Dict, Any, List, Optional, Union
import sqlalchemy
from sqlalchemy import create_engine, inspect, text
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import pyodbc
import pymysql
import psycopg2
import sqlite3
from .base_agent import BaseAgent

class DatabaseAgent(BaseAgent):
    """Agent responsible for database operations and schema discovery."""
    
    SUPPORTED_DRIVERS = {
        "mssql": "ODBC Driver 17 for SQL Server",
        "mysql": "pymysql",
        "postgresql": "psycopg2",
        "sqlite": "sqlite3",
        "oracle": "cx_oracle"
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("database_agent")
        self.config = config
        self.engine = None
        self.inspector = None
        
        # Create cache directory
        self.cache_dir = Path(config.get("cache_dir", "./cache/db"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize the database connection."""
        try:
            connection_string = self._build_connection_string()
            self.engine = create_engine(connection_string)
            self.inspector = inspect(self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info(f"Successfully connected to {self.config['type']} database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process database-related requests."""
        action = input_data.get("action")
        
        if not action:
            raise ValueError("Action not specified in input data")
        
        if not self.engine:
            raise Exception("Database not connected")
        
        try:
            if action == "get_schema_info":
                return await self._get_schema_info()
            
            elif action == "execute_query":
                query = input_data.get("query")
                params = input_data.get("params")
                return await self._execute_query(query, params)
            
            elif action == "get_table_sample":
                table_name = input_data.get("table_name")
                limit = input_data.get("limit", 1000)
                return await self._get_table_sample(table_name, limit)
            
            elif action == "get_table_stats":
                table_name = input_data.get("table_name")
                return await self._get_table_stats(table_name)
            
            elif action == "test_connection":
                return await self._test_connection()
            
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing database request: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up database resources."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.inspector = None
            self.logger.info("Database connection closed")
    
    def _build_connection_string(self) -> str:
        """Build connection string based on database type."""
        db_type = self.config["type"].lower()
        
        if db_type == "mssql":
            params = {
                "driver": self.SUPPORTED_DRIVERS["mssql"],
                "server": self.config["server"],
                "database": self.config["database"],
                "trusted_connection": self.config.get("trusted_connection", "yes"),
                "encrypt": self.config.get("encrypt", "yes")
            }
            
            if not self.config.get("trusted_connection"):
                params["uid"] = self.config["username"]
                params["pwd"] = self.config["password"]
            
            return "mssql+pyodbc:///?odbc_connect=" + ";".join(f"{k}={v}" for k, v in params.items())
            
        elif db_type == "mysql":
            return (
                f"mysql+pymysql://{self.config['username']}:{self.config['password']}"
                f"@{self.config['server']}/{self.config['database']}"
            )
            
        elif db_type == "postgresql":
            return (
                f"postgresql+psycopg2://{self.config['username']}:{self.config['password']}"
                f"@{self.config['server']}/{self.config['database']}"
            )
            
        elif db_type == "sqlite":
            return f"sqlite:///{self.config['database']}"
            
        elif db_type == "oracle":
            return (
                f"oracle+cx_oracle://{self.config['username']}:{self.config['password']}"
                f"@{self.config['server']}:{self.config.get('port', 1521)}/?service_name={self.config['service_name']}"
            )
            
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    async def _get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information."""
        schema_info = {
            "tables": {},
            "views": {},
            "relationships": [],
            "procedures": [],
            "functions": []
        }
        
        # Get table information
        for table_name in self.inspector.get_table_names():
            schema_info["tables"][table_name] = {
                "columns": self._get_column_info(table_name),
                "indexes": self._get_index_info(table_name),
                "constraints": self._get_constraint_info(table_name)
            }
        
        # Get view information
        for view_name in self.inspector.get_view_names():
            schema_info["views"][view_name] = {
                "columns": self._get_column_info(view_name),
                "definition": self._get_view_definition(view_name)
            }
        
        # Get relationships
        schema_info["relationships"] = self._get_relationships()
        
        return schema_info
    
    def _get_column_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get detailed column information for a table."""
        columns = []
        for col in self.inspector.get_columns(table_name):
            column_info = {
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "default": col.get("default"),
                "primary_key": col.get("primary_key", False)
            }
            columns.append(column_info)
        return columns
    
    def _get_index_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get index information for a table."""
        indexes = []
        for idx in self.inspector.get_indexes(table_name):
            index_info = {
                "name": idx["name"],
                "columns": idx["column_names"],
                "unique": idx.get("unique", False)
            }
            indexes.append(index_info)
        return indexes
    
    def _get_constraint_info(self, table_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get constraint information for a table."""
        constraints = {
            "primary_key": [],
            "foreign_keys": [],
            "unique": [],
            "check": []
        }
        
        # Get primary key
        pk = self.inspector.get_pk_constraint(table_name)
        if pk and pk.get("constrained_columns"):
            constraints["primary_key"].append({
                "name": pk.get("name"),
                "columns": pk["constrained_columns"]
            })
        
        # Get foreign keys
        for fk in self.inspector.get_foreign_keys(table_name):
            constraints["foreign_keys"].append({
                "name": fk.get("name"),
                "columns": fk["constrained_columns"],
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"]
            })
        
        return constraints
    
    def _get_view_definition(self, view_name: str) -> Optional[str]:
        """Get the SQL definition of a view."""
        try:
            return self.inspector.get_view_definition(view_name)
        except Exception as e:
            self.logger.warning(f"Could not get view definition for {view_name}: {str(e)}")
            return None
    
    def _get_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships between tables."""
        relationships = []
        for table_name in self.inspector.get_table_names():
            for fk in self.inspector.get_foreign_keys(table_name):
                relationship = {
                    "source_table": table_name,
                    "source_columns": fk["constrained_columns"],
                    "target_table": fk["referred_table"],
                    "target_columns": fk["referred_columns"],
                    "name": fk.get("name")
                }
                relationships.append(relationship)
        return relationships
    
    async def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        try:
            start_time = datetime.now()
            
            if params:
                result = pd.read_sql_query(text(query), self.engine, params=params)
            else:
                result = pd.read_sql_query(text(query), self.engine)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Query executed in {duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise
    
    async def _get_table_sample(self, table_name: str, limit: int = 1000) -> pd.DataFrame:
        """Get a sample of records from a table."""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return await self._execute_query(query)
    
    async def _get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get basic statistics for a table."""
        stats = {
            "row_count": None,
            "size_bytes": None,
            "column_stats": {}
        }
        
        try:
            # Get row count
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = await self._execute_query(query)
            stats["row_count"] = result.iloc[0]["count"]
            
            # Get column statistics
            sample = await self._get_table_sample(table_name)
            stats["column_stats"] = sample.describe().to_dict()
            
        except Exception as e:
            self.logger.error(f"Error getting table stats: {str(e)}")
        
        return stats
    
    async def _test_connection(self) -> Dict[str, Any]:
        """Test database connection and return status information."""
        status = {
            "success": False,
            "message": "",
            "timestamp": datetime.now().isoformat(),
            "database_type": self.config["type"],
            "server": self.config["server"],
            "database": self.config["database"]
        }
        
        try:
            if not self.engine:
                await self.initialize()
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            status["success"] = True
            status["message"] = "Connection successful"
            
        except Exception as e:
            status["message"] = f"Connection failed: {str(e)}"
        
        return status 