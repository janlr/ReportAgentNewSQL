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
        super().__init__("database_agent")
        self.config = config
        self.engine = None
        self.inspector = None
        self.cache_dir = Path(config.get("cache_dir", "./cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize database connection and inspector."""
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
            
            if action == "get_schema_info":
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
            host = self.config.get("host", "localhost")
            database = self.config.get("database", "AdventureWorks2017")
            return f"mssql+pyodbc://{host}/{database}?driver={driver}"
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
        try:
            schema_info = {
                "tables": {},
                "relationships": []
            }
            
            # Get all schemas
            with self.engine.connect() as conn:
                schemas_query = """
                SELECT DISTINCT s.name AS schema_name
                FROM sys.schemas s
                JOIN sys.tables t ON t.schema_id = s.schema_id
                ORDER BY s.name;
                """
                result = conn.execute(text(schemas_query))
                schemas = [row[0] for row in result]
            
            # Get table information for each schema
            for schema in schemas:
                with self.engine.connect() as conn:
                    tables_query = f"""
                    SELECT 
                        t.name AS table_name,
                        s.name AS schema_name,
                        OBJECT_ID(s.name + '.' + t.name) as object_id
                    FROM sys.tables t
                    JOIN sys.schemas s ON t.schema_id = s.schema_id
                    WHERE s.name = :schema
                    ORDER BY t.name;
                    """
                    
                    tables = conn.execute(text(tables_query), {"schema": schema}).fetchall()
                    
                    for table in tables:
                        full_table_name = f"{table.schema_name}.{table.table_name}"
                        
                        # Get column information in a new connection
                        with self.engine.connect() as col_conn:
                            columns_query = """
                            SELECT 
                                c.name AS column_name,
                                t.name AS data_type,
                                c.max_length,
                                c.precision,
                                c.scale,
                                c.is_nullable,
                                c.is_identity,
                                CASE WHEN pk.column_id IS NOT NULL THEN 1 ELSE 0 END AS is_primary_key
                            FROM sys.columns c
                            JOIN sys.types t ON c.user_type_id = t.user_type_id
                            LEFT JOIN (
                                SELECT ic.column_id, ic.object_id
                                FROM sys.index_columns ic
                                JOIN sys.indexes i ON ic.object_id = i.object_id 
                                    AND ic.index_id = i.index_id
                                WHERE i.is_primary_key = 1
                            ) pk ON pk.column_id = c.column_id 
                                AND pk.object_id = c.object_id
                            WHERE c.object_id = :object_id
                            ORDER BY c.column_id;
                            """
                            
                            columns = []
                            column_results = col_conn.execute(text(columns_query), {"object_id": table.object_id}).fetchall()
                            for col in column_results:
                                columns.append({
                                    "name": col.column_name,
                                    "type": col.data_type,
                                    "max_length": col.max_length,
                                    "precision": col.precision,
                                    "scale": col.scale,
                                    "nullable": col.is_nullable,
                                    "is_identity": col.is_identity,
                                    "is_primary_key": col.is_primary_key
                                })
                        
                        # Get foreign key information in a new connection
                        with self.engine.connect() as fk_conn:
                            fk_query = """
                            SELECT 
                                fk.name AS fk_name,
                                OBJECT_SCHEMA_NAME(fk.parent_object_id) AS schema_name,
                                OBJECT_NAME(fk.parent_object_id) AS table_name,
                                c1.name AS column_name,
                                OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS referenced_schema_name,
                                OBJECT_NAME(fk.referenced_object_id) AS referenced_table_name,
                                c2.name AS referenced_column_name
                            FROM sys.foreign_keys fk
                            JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                            JOIN sys.columns c1 ON fkc.parent_object_id = c1.object_id 
                                AND fkc.parent_column_id = c1.column_id
                            JOIN sys.columns c2 ON fkc.referenced_object_id = c2.object_id 
                                AND fkc.referenced_column_id = c2.column_id
                            WHERE fk.parent_object_id = :object_id;
                            """
                            
                            foreign_keys = []
                            fk_results = fk_conn.execute(text(fk_query), {"object_id": table.object_id}).fetchall()
                            for fk in fk_results:
                                foreign_keys.append({
                                    "name": fk.fk_name,
                                    "column": fk.column_name,
                                    "referenced_schema": fk.referenced_schema_name,
                                    "referenced_table": fk.referenced_table_name,
                                    "referenced_column": fk.referenced_column_name
                                })
                        
                        schema_info["tables"][full_table_name] = {
                            "name": table.table_name,
                            "schema": table.schema_name,
                            "columns": columns,
                            "foreign_keys": foreign_keys
                        }
            
            return {
                "success": True,
                "data": schema_info
            }
            
        except Exception as e:
            self.logger.error(f"Error getting schema info: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
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
                        # Add type handling for problematic columns
                        query = f"""
                        SET ARITHABORT ON;
                        SET ANSI_WARNINGS ON;
                        {query}
                        """
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
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get database connection parameters from config."""
        from . import get_config
        return get_config('database')

    async def discover_schema(self) -> Dict[str, Any]:
        """Discover database schema and suggest mappings."""
        try:
            schema_info = {}
            with self.engine.connect() as conn:
                # Get all tables
                tables = self.inspector.get_table_names()
                
                for table in tables:
                    try:
                        columns = self.inspector.get_columns(table)
                        primary_keys = self.inspector.get_pk_constraint(table)
                        foreign_keys = self.inspector.get_foreign_keys(table)
                        
                        # Ensure columns is a list of dictionaries with 'name' and 'type'
                        formatted_columns = []
                        for col in columns:
                            if isinstance(col, dict):
                                # Convert problematic SQL types to supported ones
                                col_type = str(col['type'])
                                if 'geometry' in col_type.lower():
                                    col_type = 'VARCHAR'
                                elif 'hierarchyid' in col_type.lower():
                                    col_type = 'VARCHAR'
                                elif 'geography' in col_type.lower():
                                    col_type = 'VARCHAR'
                                
                                formatted_columns.append({
                                    'name': col['name'],
                                    'type': col_type,
                                    'nullable': col.get('nullable', True),
                                    'default': col.get('default', None)
                                })
                            else:
                                formatted_columns.append({
                                    'name': str(col),
                                    'type': 'VARCHAR'  # Default to VARCHAR for unknown types
                                })
                        
                        schema_info[table] = {
                            "columns": formatted_columns,
                            "primary_keys": primary_keys,
                            "foreign_keys": foreign_keys,
                            "suggested_type": self._suggest_table_type(table, formatted_columns)
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Error processing table {table}: {str(e)}")
                        continue
                
                return schema_info
                
        except Exception as e:
            self.logger.error(f"Error discovering schema: {str(e)}")
            raise

    def _suggest_table_type(self, table_name: str, columns: List[Dict]) -> str:
        """Suggest table type based on name and structure."""
        table_lower = table_name.lower()
        if any(word in table_lower for word in ["order", "sale"]):
            return "orders"
        elif any(word in table_lower for word in ["product", "item"]):
            return "products"
        elif any(word in table_lower for word in ["customer", "client"]):
            return "customers"
        return "unknown"

    def validate_schema_mapping(self, mapping: Dict[str, Any]) -> bool:
        """Validate schema mapping against actual database schema."""
        try:
            schema_info = self.discover_schema()
            
            # Validate tables exist
            for table_key, table_name in mapping['sales'].items():
                if table_key.endswith('_table') and table_name not in schema_info:
                    self.logger.error(f"Table {table_name} not found in database")
                    return False
            
            # Validate columns exist
            for table_name in schema_info:
                columns = schema_info[table_name]['columns']
                for col_key, col_name in mapping['sales']['column_mapping'].items():
                    if col_name not in columns:
                        self.logger.error(f"Column {col_name} not found in table {table_name}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating schema mapping: {str(e)}")
            return False

    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a SQL query with enhanced type handling."""
        try:
            with self.engine.connect() as conn:
                # Execute with parameter binding if provided
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                # Get column types from result
                column_types = result.keys()
                self.logger.info("Column types in result:")
                for col in column_types:
                    self.logger.info(f"Column: {col}")
                
                # Convert results to strings safely
                rows = []
                for db_row in result.fetchall():
                    row_dict = {}
                    for i, (col, val) in enumerate(zip(column_types, db_row)):
                        try:
                            if val is None:
                                row_dict[col] = None
                            elif isinstance(val, (bytes, bytearray)):
                                row_dict[col] = '<BINARY_DATA>'
                            elif hasattr(val, 'STAsText'):  # Spatial type
                                row_dict[col] = val.STAsText()
                            else:
                                row_dict[col] = str(val)
                        except Exception as e:
                            self.logger.error(f"Error converting column {col} (index {i}): {str(e)}")
                            row_dict[col] = '<CONVERSION_ERROR>'
                    rows.append(row_dict)
                
                return {
                    "success": True,
                    "data": rows,
                    "columns": list(column_types)
                }
                
        except Exception as e:
            self.logger.error(f"Query execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _build_sample_query(self, table_name: str, limit: int = 10) -> str:
        """Build a safe sample query with type handling."""
        try:
            # Get column information and log it for debugging
            columns = self.inspector.get_columns(table_name)
            self.logger.info(f"Table {table_name} columns:")
            for i, col in enumerate(columns):
                self.logger.info(f"Column {i}: {col['name']}, Type: {col['type']}")
            
            type_mapping = self.config.get('type_mapping', {})
            unsupported_types = type_mapping.get('unsupported_types', [])
            type_handlers = type_mapping.get('type_handlers', {})
            conversion_rules = type_mapping.get('type_conversion_rules', {})
            
            # Build column list with type handling
            column_expressions = []
            for col in columns:
                try:
                    col_name = col['name']
                    col_type = str(col['type']).lower()
                    
                    # Special handling for spatial types
                    if 'geometry' in col_type:
                        expr = f"CASE WHEN {col_name} IS NULL THEN 'NULL' ELSE {col_name}.STAsText() END AS {col_name}"
                    elif 'geography' in col_type:
                        expr = f"CASE WHEN {col_name} IS NULL THEN 'NULL' ELSE {col_name}.STAsText() END AS {col_name}"
                    elif 'hierarchyid' in col_type:
                        expr = f"CASE WHEN {col_name} IS NULL THEN 'NULL' ELSE CAST({col_name} AS nvarchar(max)) END AS {col_name}"
                    else:
                        # Check if type needs conversion
                        needs_conversion = False
                        for unsupported in unsupported_types:
                            if unsupported.lower() in col_type:
                                needs_conversion = True
                                break
                        
                        if needs_conversion:
                            # Use configured type handler or default to VARCHAR(MAX)
                            conversion_type = type_handlers.get(col_type, 'VARCHAR(MAX)')
                            
                            # Apply conversion rules
                            if conversion_rules.get('handle_nulls', True):
                                expr = f"ISNULL(CAST({col_name} AS {conversion_type}), 'NULL') AS {col_name}"
                            else:
                                expr = f"CAST({col_name} AS {conversion_type}) AS {col_name}"
                        else:
                            expr = f"TRY_CAST({col_name} AS VARCHAR(MAX)) AS {col_name}"
                    
                    column_expressions.append(expr)
                    
                except Exception as e:
                    self.logger.error(f"Error processing column {col.get('name', 'unknown')}: {str(e)}")
                    # Add a placeholder for failed columns
                    column_expressions.append(f"'CONVERSION_ERROR' AS {col.get('name', 'unknown_column')}")
            
            # Build the final query with error handling and optimization hints
            query = f"""
            SET ARITHABORT ON;
            SET ANSI_WARNINGS ON;
            SET ANSI_NULLS ON;
            SET NOCOUNT ON;
            
            -- Add query hint for better performance
            SELECT TOP {limit} 
                {', '.join(column_expressions)}
            FROM {table_name}
            WITH (NOLOCK)
            OPTION (MAXDOP 1, FAST {limit})
            """
            
            return query
            
        except Exception as e:
            self.logger.error(f"Error building sample query: {str(e)}")
            raise

    async def _get_column_details(self, table_name: str) -> List[Dict[str, Any]]:
        """Get detailed column information from SQL Server."""
        query = """
        SELECT 
            c.name AS column_name,
            t.name AS data_type,
            c.max_length,
            c.precision,
            c.scale,
            c.is_nullable,
            CASE WHEN t.name IN ('geometry', 'geography', 'hierarchyid') THEN 1 ELSE 0 END as is_spatial
        FROM sys.columns c
        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
        WHERE OBJECT_ID = OBJECT_ID(?)
        ORDER BY c.column_id
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), [table_name])
                columns = [dict(row) for row in result]
                self.logger.info(f"Detailed column info for {table_name}:")
                for col in columns:
                    self.logger.info(str(col))
                return columns
        except Exception as e:
            self.logger.error(f"Error getting column details: {str(e)}")
            return []

    async def _get_problematic_columns(self, table_name: str) -> Dict[str, str]:
        """Identify columns with problematic types."""
        query = """
        SELECT 
            c.name AS column_name,
            t.name AS type_name,
            c.column_id
        FROM sys.columns c
        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
        WHERE OBJECT_ID = OBJECT_ID(@table_name)
        AND t.name IN ('geometry', 'geography', 'hierarchyid')
        ORDER BY c.column_id;
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'table_name': table_name})
                columns = {row['column_name']: row['type_name'] for row in result}
                if columns:
                    self.logger.warning(f"Found problematic columns in {table_name}: {columns}")
                return columns
        except Exception as e:
            self.logger.error(f"Error checking problematic columns: {str(e)}")
            return {}

    def _setup_logging(self):
        """Configure logging for database agent."""
        # Create a StreamHandler that writes to StringIO
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        
        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add the handler to this agent's logger
        self.logger.addHandler(handler)
        
        return log_stream

    async def get_sample_data(self, table_name: str, limit: int = 10) -> Dict[str, Any]:
        """Get sample data from a table with type handling."""
        log_stream = self._setup_logging()
        
        try:
            self.logger.info(f"Starting get_sample_data for table: {table_name}")
            
            # Parse schema and table name
            if '.' in table_name:
                schema, table = table_name.split('.')
            else:
                schema = 'dbo'
                table = table_name
            
            # Clean and escape names
            schema = schema.replace('[', '').replace(']', '')
            table = table.replace('[', '').replace(']', '')
            full_table_name = f"[{schema}].[{table}]"
            
            self.logger.info(f"Using full table name: {full_table_name}")
            
            # Get table object ID first
            object_id_query = f"""
            SELECT OBJECT_ID('{schema}.{table}') as object_id;
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(object_id_query)).fetchone()
                if not result or not result[0]:
                    return {
                        "success": False,
                        "error": f"Table {full_table_name} not found",
                        "logs": log_stream.getvalue()
                    }
                object_id = result[0]
                
                # Get column information directly from SQL Server
                columns_query = """
                SELECT 
                    c.name AS column_name,
                    t.name AS data_type,
                    c.max_length,
                    c.precision,
                    c.scale
                FROM sys.columns c
                JOIN sys.types t ON c.user_type_id = t.user_type_id
                WHERE c.object_id = :object_id
                ORDER BY c.column_id;
                """
                
                # Build column expressions with type handling
                column_expressions = []
                for col in conn.execute(text(columns_query), {'object_id': object_id}).fetchall():
                    col_name = col.column_name
                    data_type = col.data_type.lower()
                    
                    if data_type == 'uniqueidentifier':
                        # Handle GUID/uniqueidentifier type (like rowguid)
                        expr = f"CAST([{col_name}] AS CHAR(36)) AS [{col_name}]"
                    elif data_type == 'geography':
                        # Handle geography type (like SpatialLocation)
                        expr = f"""
                        CASE 
                            WHEN [{col_name}] IS NULL THEN NULL 
                            ELSE CONCAT(
                                'POINT (',
                                CAST([{col_name}].Long AS NVARCHAR(20)),
                                ' ',
                                CAST([{col_name}].Lat AS NVARCHAR(20)),
                                ')'
                            )
                        END AS [{col_name}]
                        """
                    elif data_type == 'geometry':
                        # Handle geometry type
                        expr = f"CASE WHEN [{col_name}] IS NULL THEN NULL ELSE [{col_name}].STAsText() END AS [{col_name}]"
                    elif data_type == 'hierarchyid':
                        # Handle hierarchyid
                        expr = f"CASE WHEN [{col_name}] IS NULL THEN NULL ELSE CAST([{col_name}] AS NVARCHAR(MAX)) END AS [{col_name}]"
                    else:
                        # Default handling for other types
                        expr = f"[{col_name}]"
                    
                    column_expressions.append(expr)
                
                if not column_expressions:
                    return {
                        "success": False,
                        "error": "No valid columns to query",
                        "logs": log_stream.getvalue()
                    }
                
                # Build and execute the final query
                query = f"""
                SELECT TOP {limit} 
                    {', '.join(column_expressions)}
                FROM {full_table_name}
                """
                
                self.logger.info(f"Executing query:\n{query}")
                
                try:
                    result = conn.execute(text(query))
                    rows = []
                    for row in result:
                        row_dict = {}
                        for col in result.keys():
                            try:
                                val = getattr(row, col)
                                if val is None:
                                    row_dict[col] = None
                                elif isinstance(val, bytes):
                                    # Handle binary data
                                    row_dict[col] = val.hex()
                                else:
                                    row_dict[col] = str(val)
                            except Exception as e:
                                self.logger.error(f"Error converting column {col}: {str(e)}")
                                row_dict[col] = None
                        rows.append(row_dict)
                    
                    return {
                        "success": True,
                        "data": rows,
                        "columns": list(result.keys()),
                        "total_columns": len(column_expressions),
                        "included_columns": len(column_expressions),
                        "logs": log_stream.getvalue()
                    }
                    
                except Exception as e:
                    self.logger.error(f"Query execution error: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "logs": log_stream.getvalue()
                    }
                
        except Exception as e:
            self.logger.error(f"Error in get_sample_data: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "logs": log_stream.getvalue()
            }
        finally:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler) 