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
import streamlit as st
from io import StringIO

class DatabaseAgent(BaseAgent):
    """Agent responsible for database operations, schema discovery, and relationship analysis."""
    
    SUPPORTED_DRIVERS = {
        "mssql": "ODBC Driver 17 for SQL Server",
        "mysql": "pymysql",
        "postgresql": "psycopg2",
        "sqlite": "sqlite3",
        "oracle": "cx_oracle"
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        super().__init__("database_agent", config)  # Pass config to BaseAgent
        self.engine = None
        self.inspector = None
        self.cache_dir = Path(config.get("cache_dir", "./cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set base required fields
        self.required_config = ["type", "host", "database", "driver"]
        
        # Add authentication fields if not using Windows Authentication
        if config.get("trusted_connection", "yes").lower() != "yes":
            self.required_config.extend(["user", "password"])
    
    async def initialize(self) -> bool:
        """Initialize database connection and inspector."""
        if not self.validate_config(self.required_config):
            return False
            
        try:
            self.logger.info("Building connection string...")
            connection_string = self._build_connection_string()
            
            self.logger.info("Creating database engine...")
            self.engine = create_engine(connection_string, echo=self.config.get("echo", False))
            
            self.logger.info("Creating inspector...")
            self.inspector = inspect(self.engine)
            
            # Test connection
            self.logger.info("Testing connection...")
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info("Database connection initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing database connection: {str(e)}", exc_info=True)
            return False
    
    async def cleanup(self) -> bool:
        """Clean up database resources."""
        try:
            if self.engine:
                self.engine.dispose()
            self.logger.info("Database connection cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up database connection: {str(e)}")
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process database operations."""
        try:
            if not self.validate_input(input_data, ["action"]):
                raise ValueError("Missing required field: action")
            
            action = input_data["action"]
            parameters = input_data.get("parameters", {})
            
            if action == "execute_query":
                query = input_data.get("query")
                params = input_data.get("params", {})
                
                if not query:
                    raise ValueError("Query is required for execute_query action")
                
                # Execute query and return results as DataFrame
                with self.engine.connect() as conn:
                    result = pd.read_sql(text(query), conn, params=params)
                
                return {
                    "success": True,
                    "data": result
                }
            
            elif action == "get_schema_info":
                return await self._get_schema_info()
            
            elif action == "get_product_categories":
                return await self._get_product_categories()
            
            elif action == "get_sample_data":
                table_name = parameters.get("table_name")
                limit = parameters.get("limit", 10)
                if not table_name:
                    raise ValueError("Missing required parameter: table_name")
                return await self.get_sample_data(table_name, limit)
            
            elif action == "analyze_relationships":
                # Analyze relationships between tables
                relationships = self._analyze_relationships()
                return {
                    "success": True,
                    "data": relationships
                }
            
            elif action == "suggest_joins":
                source_table = input_data.get("source_table")
                target_table = input_data.get("target_table")
                
                if not source_table or not target_table:
                    raise ValueError("Source and target tables are required")
                
                join_suggestions = self._suggest_joins(source_table, target_table)
                return {
                    "success": True,
                    "data": join_suggestions
                }
            
            elif action == "validate_query":
                query = input_data.get("query")
                if not query:
                    raise ValueError("Query is required")
                
                validation_result = self._validate_query(query)
                return {
                    "success": True,
                    "data": validation_result
                }
            
            elif action == "get_table_sample":
                table_name = input_data.get("table_name")
                limit = input_data.get("limit", 1000)
                return await self._get_table_sample(table_name, limit)
            
            elif action == "get_table_stats":
                table_name = input_data.get("table_name")
                return await self._get_table_stats(table_name)
            
            elif action == "test_connection":
                return await self._test_connection()
            
            elif action == "get_warehouse_locations":
                return await self._get_warehouse_locations()
            
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing database operation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_connection_string(self) -> str:
        """Build database connection string from config."""
        db_type = self.config.get("type", "mssql").lower()
        
        if db_type == "mssql":
            driver = self.config.get("driver", self.SUPPORTED_DRIVERS["mssql"])
            host = self.config.get("host", "localhost").replace('\\\\', '\\')  # Handle escaped backslashes
            database = self.config.get("database")
            
            # Check if using Windows Authentication
            if self.config.get("trusted_connection", "yes").lower() == "yes":
                return (
                    f"mssql+pyodbc://@{host}/{database}?"
                    f"driver={driver}&trusted_connection=yes"
                )
            else:
                # Fall back to SQL Server Authentication if explicitly configured
                user = self.config.get("user")
                password = self.config.get("password")
                port = self.config.get("port", 1433)
                return f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver}"
                
        elif db_type == "mysql":
            connection_string = (
                f"mysql+pymysql://{self.config['username']}:{self.config['password']}"
                f"@{self.config['server']}/{self.config['database']}"
            )
        elif db_type == "postgresql":
            connection_string = (
                f"postgresql+psycopg2://{self.config['username']}:{self.config['password']}"
                f"@{self.config['server']}/{self.config['database']}"
            )
        elif db_type == "sqlite":
            connection_string = f"sqlite:///{self.config['database']}"
        elif db_type == "oracle":
            connection_string = (
                f"oracle+cx_oracle://{self.config['username']}:{self.config['password']}"
                f"@{self.config['server']}:{self.config.get('port', 1521)}/?service_name={self.config['service_name']}"
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
            
        return connection_string
    
    async def _get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information."""
        try:
            schema_info = {
                "tables": {},
                "views": {},
                "relationships": []
            }
            
            with self.engine.connect() as conn:
                # Get tables and views
                tables_query = """
                SELECT 
                    t.TABLE_SCHEMA as schema_name,
                    t.TABLE_NAME as table_name,
                    t.TABLE_TYPE as table_type
                FROM INFORMATION_SCHEMA.TABLES t
                ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME
                """
                tables = pd.read_sql(text(tables_query), conn)
                
                # Get columns for each table
                for _, row in tables.iterrows():
                    schema = row["schema_name"]
                    table = row["table_name"]
                    table_type = row["table_type"]
                    
                    columns_query = f"""
                    SELECT 
                        c.COLUMN_NAME as name,
                        c.DATA_TYPE as type,
                        c.IS_NULLABLE as nullable,
                        CASE WHEN kcu.COLUMN_NAME IS NOT NULL THEN 'YES' ELSE 'NO' END as is_primary_key
                    FROM INFORMATION_SCHEMA.COLUMNS c
                    LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                        ON c.TABLE_SCHEMA = kcu.TABLE_SCHEMA
                        AND c.TABLE_NAME = kcu.TABLE_NAME
                        AND c.COLUMN_NAME = kcu.COLUMN_NAME
                        AND kcu.CONSTRAINT_NAME LIKE 'PK%'
                    WHERE c.TABLE_SCHEMA = '{schema}'
                        AND c.TABLE_NAME = '{table}'
                    ORDER BY c.ORDINAL_POSITION
                    """
                    columns = pd.read_sql(text(columns_query), conn)
                    
                    table_info = {
                        "schema": schema,
                        "name": table,
                        "type": table_type,
                        "columns": columns.to_dict("records")
                    }
                    
                    if table_type == "BASE TABLE":
                        schema_info["tables"][f"{schema}.{table}"] = table_info
                    else:
                        schema_info["views"][f"{schema}.{table}"] = table_info
                
                # Get relationships
                relationships_query = """
                SELECT 
                    fk.name as constraint_name,
                    OBJECT_SCHEMA_NAME(fk.parent_object_id) as parent_schema,
                    OBJECT_NAME(fk.parent_object_id) as parent_table,
                    c1.name as parent_column,
                    OBJECT_SCHEMA_NAME(fk.referenced_object_id) as referenced_schema,
                    OBJECT_NAME(fk.referenced_object_id) as referenced_table,
                    c2.name as referenced_column
                FROM sys.foreign_keys fk
                INNER JOIN sys.foreign_key_columns fkc 
                    ON fk.object_id = fkc.constraint_object_id
                INNER JOIN sys.columns c1
                    ON fkc.parent_object_id = c1.object_id
                    AND fkc.parent_column_id = c1.column_id
                INNER JOIN sys.columns c2
                    ON fkc.referenced_object_id = c2.object_id
                    AND fkc.referenced_column_id = c2.column_id
                ORDER BY parent_schema, parent_table
                """
                relationships = pd.read_sql(text(relationships_query), conn)
                schema_info["relationships"] = relationships.to_dict("records")
            
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Error getting schema information: {str(e)}")
            return {}
    
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
    
    def _analyze_relationships(self) -> List[Dict[str, Any]]:
        """Analyze relationships between tables."""
        try:
            relationships = []
            
            with self.engine.connect() as conn:
                # Get all foreign key relationships
                query = """
                SELECT 
                    fk.name as constraint_name,
                    OBJECT_SCHEMA_NAME(fk.parent_object_id) as source_schema,
                    OBJECT_NAME(fk.parent_object_id) as source_table,
                    c1.name as source_column,
                    OBJECT_SCHEMA_NAME(fk.referenced_object_id) as target_schema,
                    OBJECT_NAME(fk.referenced_object_id) as target_table,
                    c2.name as target_column,
                    'FOREIGN KEY' as relationship_type
                FROM sys.foreign_keys fk
                INNER JOIN sys.foreign_key_columns fkc 
                    ON fk.object_id = fkc.constraint_object_id
                INNER JOIN sys.columns c1
                    ON fkc.parent_object_id = c1.object_id
                    AND fkc.parent_column_id = c1.column_id
                INNER JOIN sys.columns c2
                    ON fkc.referenced_object_id = c2.object_id
                    AND fkc.referenced_column_id = c2.column_id
                """
                result = pd.read_sql(text(query), conn)
                relationships.extend(result.to_dict("records"))
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error analyzing relationships: {str(e)}")
            return []
    
    def _suggest_joins(self, source_table: str, target_table: str) -> List[Dict[str, Any]]:
        """Suggest possible join paths between two tables."""
        try:
            suggestions = []
            relationships = self._analyze_relationships()
            
            # Direct relationships
            direct_joins = [
                rel for rel in relationships
                if (f"{rel['source_schema']}.{rel['source_table']}" == source_table and 
                    f"{rel['target_schema']}.{rel['target_table']}" == target_table) or
                   (f"{rel['source_schema']}.{rel['source_table']}" == target_table and 
                    f"{rel['target_schema']}.{rel['target_table']}" == source_table)
            ]
            
            if direct_joins:
                suggestions.extend([{
                    "path_type": "direct",
                    "joins": [join]
                } for join in direct_joins])
            
            # Indirect relationships (one intermediate table)
            for rel1 in relationships:
                for rel2 in relationships:
                    if (f"{rel1['source_schema']}.{rel1['source_table']}" == source_table and
                        f"{rel2['target_schema']}.{rel2['target_table']}" == target_table and
                        f"{rel1['target_schema']}.{rel1['target_table']}" == f"{rel2['source_schema']}.{rel2['source_table']}"):
                        suggestions.append({
                            "path_type": "indirect",
                            "intermediate_table": f"{rel1['target_schema']}.{rel1['target_table']}",
                            "joins": [rel1, rel2]
                        })
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error suggesting joins: {str(e)}")
            return []
    
    def _validate_query(self, query: str) -> Dict[str, Any]:
        """Validate a SQL query."""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            with self.engine.connect() as conn:
                try:
                    # Try to compile the query
                    compiled_query = text(query).compile(bind=self.engine)
                    
                    # Try to execute the query with EXPLAIN/EXECUTION PLAN
                    if self.config.get("type") == "mssql":
                        explain_query = f"SET SHOWPLAN_XML ON; {query}"
                        conn.execute(text(explain_query))
                    
                except Exception as e:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(str(e))
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating query: {str(e)}")
            return {
                "is_valid": False,
                "errors": [str(e)],
                "warnings": []
            }
    
    async def _get_product_categories(self) -> Dict[str, Any]:
        """Get list of product categories."""
        # TODO: Implement actual category retrieval
        return {
            "success": True,
            "data": []
        }
    
    async def _get_warehouse_locations(self) -> Dict[str, Any]:
        """Get list of warehouse locations."""
        # TODO: Implement actual location retrieval
        return {
            "success": True,
            "data": []
        }
