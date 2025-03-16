from typing import Dict, Any, List, Optional
import pyodbc
import pandas as pd
import logging
from .base_agent import BaseAgent

class DatabaseAgent(BaseAgent):
    """Agent responsible for database operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the database agent."""
        super().__init__("database_agent", config)
        self.required_config = ["host", "database", "driver"]
        self.connection = None
        self.logger = logging.getLogger(__name__)
    
    def _build_connection_string(self) -> str:
        """Build the connection string based on configuration."""
        try:
            # Start with required parameters
            conn_params = {
                "DRIVER": f"{{{self.config['driver']}}}",
                "SERVER": self.config["host"],
                "DATABASE": self.config["database"]
            }
            
            # Add authentication parameters
            if self.config.get("trusted_connection", "yes").lower() == "yes":
                conn_params["Trusted_Connection"] = "yes"
            else:
                conn_params.update({
                    "UID": self.config["user"],
                    "PWD": self.config["password"]
                })
            
            # Build the connection string
            conn_str = ";".join(f"{k}={v}" for k, v in conn_params.items())
            self.logger.info(f"Built connection string: {conn_str}")
            return conn_str
            
        except Exception as e:
            self.logger.error(f"Error building connection string: {str(e)}")
            raise
    
    async def initialize(self) -> bool:
        """Initialize database connection."""
        try:
            # Build and test connection string
            conn_str = self._build_connection_string()
            self.logger.info("Attempting to connect to database...")
            
            # Create connection
            self.connection = pyodbc.connect(conn_str)
            self.logger.info("Successfully connected to database")
            
            # Test connection by getting table list
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT TABLE_SCHEMA, TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
            """)
            tables = cursor.fetchall()
            self.logger.info(f"Found {len(tables)} tables in database")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {str(e)}")
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
                self.connection = None
            return False
    
    async def cleanup(self) -> bool:
        """Clean up database resources."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            return True
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process database operations."""
        try:
            if not self.connection:
                raise RuntimeError("Database connection not initialized")
                
            action = input_data.get("action")
            if not action:
                raise ValueError("No action specified")
            
            if action == "get_tables":
                return await self._get_tables()
            elif action == "execute_query":
                return await self._execute_query(input_data.get("query", ""), input_data.get("params", {}))
            elif action == "get_schema_info":
                return await self._get_schema_info(input_data.get("table_name"), input_data.get("schema_name", "dbo"))
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing database operation: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_tables(self) -> Dict[str, Any]:
        """Get list of tables in the database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT 
                    TABLE_SCHEMA,
                    TABLE_NAME,
                    TABLE_TYPE
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_SCHEMA, TABLE_NAME
            """)
            
            tables = [{"schema": schema, "name": name, "type": type_}
                     for schema, name, type_ in cursor.fetchall()]
            
            return {
                "success": True,
                "data": {
                    "tables": tables
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting tables: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _execute_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a SQL query."""
        try:
            if not query:
                raise ValueError("No query provided")
                
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            
            # If SELECT query, return results as DataFrame
            if query.strip().upper().startswith("SELECT"):
                columns = [column[0] for column in cursor.description]
                results = cursor.fetchall()
                
                # Convert results to list of dicts for consistent format
                data = []
                for row in results:
                    data.append(dict(zip(columns, row)))
                
                return {
                    "success": True,
                    "data": data
                }
            
            # For other queries, commit and return success
            self.connection.commit()
            return {
                "success": True,
                "data": {
                    "rows_affected": cursor.rowcount
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_schema_info(self, table_name: Optional[str], schema_name: str = "dbo") -> Dict[str, Any]:
        """Get detailed schema information for tables."""
        try:
            cursor = self.connection.cursor()
            
            # If table_name is provided, get schema for specific table
            if table_name:
                query = """
                    SELECT 
                        c.COLUMN_NAME,
                        c.DATA_TYPE,
                        c.CHARACTER_MAXIMUM_LENGTH,
                        c.NUMERIC_PRECISION,
                        c.NUMERIC_SCALE,
                        c.IS_NULLABLE,
                        CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 'YES' ELSE 'NO' END AS IS_PRIMARY_KEY,
                        CASE WHEN fk.COLUMN_NAME IS NOT NULL THEN 'YES' ELSE 'NO' END AS IS_FOREIGN_KEY,
                        fk.REFERENCED_TABLE_SCHEMA,
                        fk.REFERENCED_TABLE_NAME,
                        fk.REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS c
                    LEFT JOIN (
                        SELECT ku.TABLE_CATALOG, ku.TABLE_SCHEMA, ku.TABLE_NAME, ku.COLUMN_NAME
                        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
                            ON tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                            AND tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                            AND tc.TABLE_SCHEMA = ku.TABLE_SCHEMA
                            AND tc.TABLE_NAME = ku.TABLE_NAME
                    ) pk 
                    ON c.TABLE_SCHEMA = pk.TABLE_SCHEMA 
                    AND c.TABLE_NAME = pk.TABLE_NAME 
                    AND c.COLUMN_NAME = pk.COLUMN_NAME
                    LEFT JOIN (
                        SELECT 
                            cu.TABLE_SCHEMA,
                            cu.TABLE_NAME,
                            cu.COLUMN_NAME,
                            cu.REFERENCED_TABLE_SCHEMA,
                            cu.REFERENCED_TABLE_NAME,
                            cu.REFERENCED_COLUMN_NAME
                        FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                        JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE cu 
                            ON rc.CONSTRAINT_NAME = cu.CONSTRAINT_NAME
                            AND rc.CONSTRAINT_SCHEMA = cu.CONSTRAINT_SCHEMA
                    ) fk
                    ON c.TABLE_SCHEMA = fk.TABLE_SCHEMA
                    AND c.TABLE_NAME = fk.TABLE_NAME
                    AND c.COLUMN_NAME = fk.COLUMN_NAME
                    WHERE c.TABLE_SCHEMA = ?
                    AND c.TABLE_NAME = ?
                    ORDER BY c.ORDINAL_POSITION
                """
                cursor.execute(query, (schema_name, table_name))
                
                columns = []
                for row in cursor.fetchall():
                    column_info = {
                        "name": row[0],
                        "data_type": row[1],
                        "max_length": row[2],
                        "numeric_precision": row[3],
                        "numeric_scale": row[4],
                        "is_nullable": row[5] == "YES",
                        "is_primary_key": row[6] == "YES",
                        "is_foreign_key": row[7] == "YES"
                    }
                    
                    # Add foreign key reference information if applicable
                    if column_info["is_foreign_key"]:
                        column_info["foreign_key_reference"] = {
                            "schema": row[8],
                            "table": row[9],
                            "column": row[10]
                        }
                    
                    columns.append(column_info)
                
                return {
                    "success": True,
                    "data": {
                        "table_name": table_name,
                        "schema_name": schema_name,
                        "columns": columns
                    }
                }
            
            # If no table_name provided, get list of all tables with their column counts
            else:
                query = """
                    SELECT 
                        t.TABLE_SCHEMA,
                        t.TABLE_NAME,
                        COUNT(c.COLUMN_NAME) as COLUMN_COUNT,
                        STRING_AGG(c.COLUMN_NAME, ', ') as COLUMNS
                    FROM INFORMATION_SCHEMA.TABLES t
                    JOIN INFORMATION_SCHEMA.COLUMNS c 
                        ON t.TABLE_SCHEMA = c.TABLE_SCHEMA 
                        AND t.TABLE_NAME = c.TABLE_NAME
                    WHERE t.TABLE_TYPE = 'BASE TABLE'
                    GROUP BY t.TABLE_SCHEMA, t.TABLE_NAME
                    ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME
                """
                cursor.execute(query)
                
                tables = []
                for row in cursor.fetchall():
                    tables.append({
                        "schema": row[0],
                        "name": row[1],
                        "column_count": row[2],
                        "columns": row[3].split(", ")
                    })
                
                return {
                    "success": True,
                    "data": {
                        "tables": tables
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error getting schema information: {str(e)}")
            return {"success": False, "error": str(e)} 