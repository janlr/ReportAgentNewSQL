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
            
    # ... rest of the class methods would go here ... 