from typing import Dict, Any, List, Set, Optional, Tuple
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError, DBAPIError
import sqlalchemy
import pandas as pd
from datetime import datetime
import os
import time
from pathlib import Path
from urllib.parse import quote_plus
from .base_agent import BaseAgent

class DatabaseAgent(BaseAgent):
    """Agent responsible for database operations."""
    
    SUPPORTED_DRIVERS = {
        "mssql": "ODBC Driver 17 for SQL Server",
        "mysql": "MySQL ODBC 8.0 Unicode Driver",
        "postgres": "PostgreSQL Unicode"
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the database agent."""
        super().__init__("database_agent", config)
        self.engine = None
        self.inspector = None
        self.cache_dir = Path(config.get("cache_dir", "./cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set base required fields - remove port since it's not needed for named instances
        self.required_config = ["host", "database", "driver"]
        
        # Add authentication fields if not using Windows Authentication
        if config.get("trusted_connection", "yes").lower() != "yes":
            self.required_config.extend(["user", "password"])

    def validate_config(self, required_fields: List[str]) -> bool:
        """Validate configuration has all required fields and valid values."""
        try:
            # Check for required fields
            missing = [field for field in required_fields if field not in self.config]
            if missing:
                self.logger.error(f"Missing required configuration fields: {', '.join(missing)}")
                return False
            
            # Validate specific fields
            if 'type' in self.config and self.config['type'].lower() not in self.SUPPORTED_DRIVERS:
                self.logger.error(f"Unsupported database type: {self.config['type']}")
                return False
                
            if 'driver' in self.config:
                db_type = self.config.get('type', 'mssql').lower()
                if self.config['driver'] != self.SUPPORTED_DRIVERS.get(db_type):
                    self.logger.error(f"Invalid driver for {db_type}: {self.config['driver']}")
                    self.logger.error(f"Expected driver: {self.SUPPORTED_DRIVERS.get(db_type)}")
                    return False
            
            # Only validate port if it's provided
            if 'port' in self.config:
                try:
                    port = int(self.config['port'])
                    if port <= 0 or port > 65535:
                        self.logger.error(f"Invalid port number: {port}")
                        return False
                except (ValueError, TypeError):
                    self.logger.error(f"Port must be a valid number, got: {self.config['port']}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            return False
    
    async def initialize(self) -> bool:
        """Initialize database connection and inspector."""
        try:
            self.logger.info("Validating database configuration...")
            if not self.validate_config(self.required_config):
                raise ValueError("Invalid database configuration")
            
            self.logger.info("Building connection string...")
            connection_string = self._build_connection_string()
            if not connection_string:
                raise ValueError("Failed to build connection string")
            
            self.logger.info("Creating database engine...")
            self.engine = create_engine(
                connection_string,
                echo=self.config.get("echo", False),
                connect_args={
                    "timeout": 30
                }
            )
            
            self.logger.info("Creating inspector...")
            self.inspector = inspect(self.engine)
            
            # Test connection with detailed error handling
            self.logger.info("Testing connection...")
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT @@version as version"))
                    version = result.fetchone()[0]
                    self.logger.info(f"Connected successfully. SQL Server version: {version}")
            except Exception as conn_error:
                self.logger.error(f"Connection test failed: {str(conn_error)}")
                self.logger.error("Connection parameters:")
                self.logger.error(f"  Host: {self.config.get('host')}")
                self.logger.error(f"  Database: {self.config.get('database')}")
                self.logger.error(f"  Driver: {self.config.get('driver')}")
                self.logger.error(f"  Port: {self.config.get('port')}")
                self.logger.error(f"  Trusted Connection: {self.config.get('trusted_connection')}")
                if isinstance(conn_error, sqlalchemy.exc.DBAPIError):
                    self.logger.error(f"DBAPI error code: {conn_error.orig.args[0] if conn_error.orig.args else 'Unknown'}")
                    self.logger.error(f"DBAPI error message: {conn_error.orig.args[1] if len(conn_error.orig.args) > 1 else 'No message'}")
                raise ValueError(f"Database connection test failed: {str(conn_error)}")
            
            self.logger.info("Database connection initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}", exc_info=True)
            if isinstance(e, sqlalchemy.exc.DBAPIError):
                self.logger.error(f"DBAPI error code: {e.orig.args[0] if e.orig.args else 'Unknown'}")
                self.logger.error(f"DBAPI error message: {e.orig.args[1] if len(e.orig.args) > 1 else 'No message'}")
            raise RuntimeError(f"Failed to initialize database: {str(e)}")
    
    def _build_connection_string(self) -> str:
        """Build database connection string from config."""
        try:
            db_type = self.config.get("type", "mssql").lower()
            
            if db_type == "mssql":
                driver = self.config.get("driver", self.SUPPORTED_DRIVERS["mssql"])
                host = self.config.get("host", "localhost")
                database = self.config.get("database")
                
                self.logger.info("Connection components:")
                self.logger.info(f"  DB Type: {db_type}")
                self.logger.info(f"  Driver: {driver}")
                self.logger.info(f"  Host: {host}")
                self.logger.info(f"  Database: {database}")
                
                # Build connection string using pyodbc format
                if self.config.get("trusted_connection", "yes").lower() == "yes":
                    connection_string = (
                        "mssql+pyodbc:///?odbc_connect=" + quote_plus(
                            f"Driver={{{driver}}};"
                            f"Server={host};"
                            f"Database={database};"
                            f"Trusted_Connection=yes;"
                            f"TrustServerCertificate=yes"
                        )
                    )
                else:
                    user = self.config.get("user")
                    password = self.config.get("password")
                    if not user or not password:
                        raise ValueError("SQL Server Authentication requires both user and password")
                    connection_string = (
                        "mssql+pyodbc:///?odbc_connect=" + quote_plus(
                            f"Driver={{{driver}}};"
                            f"Server={host};"
                            f"Database={database};"
                            f"UID={user};"
                            f"PWD={password};"
                            f"TrustServerCertificate=yes"
                        )
                    )
                
                self.logger.info("Connection string built successfully")
                self.logger.debug(f"Connection string (without credentials): {connection_string.split('PWD=')[0] if 'PWD=' in connection_string else connection_string}")
                return connection_string
                
        except Exception as e:
            self.logger.error(f"Error building connection string: {str(e)}")
            raise ValueError(f"Failed to build connection string: {str(e)}")
            
    async def _get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information."""
        try:
            schema_info = {
                "tables": {},
                "views": {},
                "relationships": []
            }
            
            # Get all schemas
            schemas = self.inspector.get_schema_names()
            
            # Get tables in each schema
            for schema in schemas:
                # Skip system schemas
                if schema in ['sys', 'INFORMATION_SCHEMA', 'guest', 'db_owner', 'db_accessadmin', 
                              'db_securityadmin', 'db_ddladmin', 'db_backupoperator', 'db_datareader', 
                              'db_datawriter', 'db_denydatareader', 'db_denydatawriter']:
                    continue
                    
                # Get tables
                tables = self.inspector.get_table_names(schema=schema)
                
                # Get table details
                for table in tables:
                    full_table_name = f"{schema}.{table}"
                    
                    # Get columns
                    columns = self.inspector.get_columns(table, schema=schema)
                    formatted_columns = []
                    
                    for col in columns:
                        column_info = {
                            "name": col['name'],
                            "type": str(col['type']),
                            "nullable": col.get('nullable', True),
                            "default": col.get('default'),
                            "primary_key": col.get('primary_key', False)
                        }
                        formatted_columns.append(column_info)
                    
                    # Get primary key
                    pk_columns = []
                    try:
                        pk_constraint = self.inspector.get_pk_constraint(table, schema=schema)
                        if pk_constraint:
                            pk_columns = pk_constraint.get('constrained_columns', [])
                    except Exception as e:
                        self.logger.warning(f"Could not get PK for {full_table_name}: {str(e)}")
                    
                    # Get foreign keys
                    foreign_keys = []
                    try:
                        fk_constraints = self.inspector.get_foreign_keys(table, schema=schema)
                        for fk in fk_constraints:
                            foreign_keys.append({
                                "name": fk.get('name'),
                                "constrained_columns": fk.get('constrained_columns', []),
                                "referred_schema": fk.get('referred_schema'),
                                "referred_table": fk.get('referred_table'),
                                "referred_columns": fk.get('referred_columns', [])
                            })
                            
                            # Add to relationships list
                            schema_info["relationships"].append({
                                "source_table": full_table_name,
                                "source_columns": fk.get('constrained_columns', []),
                                "target_table": f"{fk.get('referred_schema')}.{fk.get('referred_table')}",
                                "target_columns": fk.get('referred_columns', []),
                                "name": fk.get('name')
                            })
                    except Exception as e:
                        self.logger.warning(f"Could not get FKs for {full_table_name}: {str(e)}")
                    
                    # Get indexes
                    indexes = []
                    try:
                        idx_constraints = self.inspector.get_indexes(table, schema=schema)
                        for idx in idx_constraints:
                            indexes.append({
                                "name": idx.get('name'),
                                "unique": idx.get('unique', False),
                                "columns": idx.get('column_names', [])
                            })
                    except Exception as e:
                        self.logger.warning(f"Could not get indexes for {full_table_name}: {str(e)}")
                    
                    # Store table info
                    schema_info["tables"][full_table_name] = {
                        "name": table,
                        "schema": schema,
                        "columns": formatted_columns,
                        "primary_key_columns": pk_columns,
                        "foreign_keys": foreign_keys,
                        "indexes": indexes
                    }
                
                # Get views
                views = self.inspector.get_view_names(schema=schema)
                
                # Get view details
                for view in views:
                    full_view_name = f"{schema}.{view}"
                    
                    # Get columns
                    columns = self.inspector.get_columns(view, schema=schema)
                    formatted_columns = []
                    
                    for col in columns:
                        column_info = {
                            "name": col['name'],
                            "type": str(col['type']),
                            "nullable": col.get('nullable', True)
                        }
                        formatted_columns.append(column_info)
                    
                    # Get view definition
                    view_definition = None
                    try:
                        view_definition = self.inspector.get_view_definition(view, schema=schema)
                    except Exception as e:
                        self.logger.warning(f"Could not get definition for view {full_view_name}: {str(e)}")
                    
                    # Store view info
                    schema_info["views"][full_view_name] = {
                        "name": view,
                        "schema": schema,
                        "columns": formatted_columns,
                        "definition": view_definition
                    }
            
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Error getting schema info: {str(e)}")
            raise ValueError(f"Failed to get schema information: {str(e)}")
    
    async def _test_connection(self) -> Dict[str, Any]:
        """Test database connection and return status information."""
        status = {
            "success": False,
            "message": "",
            "timestamp": datetime.now().isoformat(),
            "database_type": self.config.get("type", "mssql"),
            "server": self.config.get("host", ""),
            "database": self.config.get("database", "")
        }
        
        try:
            if not self.engine:
                self.logger.info("No engine exists, initializing...")
                await self.initialize()
            
            self.logger.info("Testing connection with simple query...")
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT @@version as version"))
                version = result.fetchone()[0]
                self.logger.info(f"Connection test successful. Version: {version}")
            
            status["success"] = True
            status["message"] = "Connection successful"
            status["version"] = version
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Connection test failed: {error_msg}")
            
            # Add detailed connection info to status
            status.update({
                "error_details": error_msg,
                "connection_info": {
                    "host": self.config.get("host"),
                    "database": self.config.get("database"),
                    "driver": self.config.get("driver"),
                    "trusted_connection": self.config.get("trusted_connection", "yes")
                }
            })
            
            if isinstance(e, sqlalchemy.exc.DBAPIError):
                status["error_code"] = e.orig.args[0] if e.orig.args else "Unknown"
                status["error_message"] = e.orig.args[1] if len(e.orig.args) > 1 else "No message"
            
            status["message"] = f"Connection failed: {error_msg}"
        
        return status
    
    async def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a SQL query and return the results as a DataFrame."""
        try:
            if not self.engine:
                await self.initialize()
            
            start_time = time.time()
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                data = []
                
                if result.returns_rows:
                    columns = result.keys()
                    data = [dict(zip(columns, row)) for row in result.fetchall()]
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Query executed in {elapsed_time:.2f} seconds")
            self.logger.debug(f"Query: {query}")
            self.logger.debug(f"Params: {params}")
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            self.logger.error(f"Query: {query}")
            self.logger.error(f"Params: {params}")
            raise ValueError(f"Failed to execute query: {str(e)}")
    
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
                raise ValueError("Invalid input data: 'action' is required")
            
            action = input_data["action"]
            
            if action == "test_connection":
                result = await self._test_connection()
                return {"success": result["success"], "data": result}
                
            elif action == "get_schema_info":
                schema_info = await self._get_schema_info()
                return {"success": True, "data": schema_info}
                
            elif action == "execute_query":
                if not self.validate_input(input_data, ["parameters.query"]):
                    raise ValueError("Invalid input data: 'parameters.query' is required")
                
                query = input_data["parameters"]["query"]
                params = input_data["parameters"].get("params")
                
                df = await self._execute_query(query, params)
                return {"success": True, "data": df.to_dict(orient="records")}
                
            elif action == "search_tables":
                if not self.validate_input(input_data, ["parameters.search_term"]):
                    raise ValueError("Invalid input data: 'parameters.search_term' is required")
                
                search_term = input_data["parameters"]["search_term"].lower()
                schema_info = await self._get_schema_info()
                
                results = []
                for table_name, table_info in schema_info["tables"].items():
                    if search_term in table_name.lower():
                        results.append(table_info)
                        continue
                    
                    # Search in column names
                    for column in table_info["columns"]:
                        if search_term in column["name"].lower():
                            results.append(table_info)
                            break
                
                return {"success": True, "data": results}
                
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            return await self.handle_error(e, {"input": input_data}) 