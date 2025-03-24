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
        
        # System tables to exclude - configurable through config
        self.system_tables = set(self.config.get("system_tables", [
            "sysdiagrams",  # Common system table for diagrams
            "tblErrorLog",  # Common system table for error logging
            "tblAuditLog"   # Common system table for audit logging
        ]))
        
        # Add any additional system tables from config
        if "additional_system_tables" in self.config:
            self.system_tables.update(self.config["additional_system_tables"])
        
        # Category configuration - should be provided in config
        self.category_config = self.config.get("category_config", {
            "category_table": {
                "name": "DimProductCategory",
                "schema": "dbo",
                "key_column": "ProductCategoryKey",
                "name_column": "EnglishProductCategoryName"
            },
            "subcategory_table": {
                "name": "DimProductSubcategory",
                "schema": "dbo",
                "key_column": "ProductSubcategoryKey",
                "name_column": "EnglishProductSubcategoryName",
                "parent_key_column": "ProductCategoryKey"
            }
        })
    
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
            elif action == "get_hierarchical_data":
                return await self._get_hierarchical_data(
                    input_data.get("parent_table"),
                    input_data.get("child_table"),
                    input_data.get("relationship_config")
                )
            elif action == "get_join_patterns":
                return await self._get_join_patterns(
                    input_data.get("parameters", {}).get("table_name"),
                    input_data.get("parameters", {}).get("schema_name", "dbo")
                )
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing database operation: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_tables(self) -> Dict[str, Any]:
        """Get list of tables in the database, excluding system tables."""
        try:
            cursor = self.connection.cursor()
            
            # Get all base tables
            cursor.execute("""
                SELECT 
                    t.TABLE_SCHEMA,
                    t.TABLE_NAME,
                    t.TABLE_TYPE,
                    COUNT(c.COLUMN_NAME) as COLUMN_COUNT,
                    STRING_AGG(c.COLUMN_NAME, ', ') as COLUMNS
                FROM INFORMATION_SCHEMA.TABLES t
                JOIN INFORMATION_SCHEMA.COLUMNS c 
                    ON t.TABLE_SCHEMA = c.TABLE_SCHEMA 
                    AND t.TABLE_NAME = c.TABLE_NAME
                WHERE t.TABLE_TYPE = 'BASE TABLE'
                AND t.TABLE_NAME NOT IN ({})
                GROUP BY t.TABLE_SCHEMA, t.TABLE_NAME, t.TABLE_TYPE
                ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME
            """.format(','.join(['?' for _ in self.system_tables])), list(self.system_tables))
            
            tables = []
            for row in cursor.fetchall():
                columns = row[4].split(", ") if row[4] else []
                tables.append({
                    "schema": row[0],
                    "name": row[1],
                    "type": row[2],
                    "column_count": row[3],
                    "columns": columns
                })
            
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
            
            # Execute query with or without parameters
            if params and len(params) > 0:
                # Check if query uses named parameters
                if '@' in query:
                    cursor.execute(query, params)
                else:
                    # For queries without named parameters, don't pass params
                    cursor.execute(query)
            else:
                cursor.execute(query)
            
            # If SELECT query, return results as DataFrame
            if query.strip().upper().startswith("SELECT"):
                columns = [column[0] for column in cursor.description]
                results = cursor.fetchall()
                
                # Convert results to list of dicts for consistent format
                data = []
                for row in results:
                    row_dict = {}
                    for i, value in enumerate(row):
                        column_name = columns[i]
                        # Handle different data types appropriately
                        if isinstance(value, (int, float)):
                            # Keep numeric values as is without formatting
                            row_dict[column_name] = value
                        elif value is None:
                            row_dict[column_name] = None
                        else:
                            # Convert other values to strings
                            row_dict[column_name] = str(value)
                    data.append(row_dict)
                
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
                # Check if table is a system table
                if table_name in self.system_tables:
            return {
                "success": False,
                        "error": f"Access to system table '{table_name}' is not allowed"
                    }
                    
                # First get basic column information
                query = """
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    CHARACTER_MAXIMUM_LENGTH,
                    IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ?
                AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
                """
                cursor.execute(query, (schema_name, table_name))
                            
                            columns = []
                for row in cursor.fetchall():
                                columns.append({
                        "name": row[0],
                        "data_type": row[1],
                        "max_length": row[2],
                        "is_nullable": "YES" if row[3] == "YES" else "NO",
                        "is_primary_key": "NO",  # Will be updated later
                        "is_foreign_key": "NO"   # Will be updated later
                    })
                
                # Get foreign key information
                try:
                            fk_query = """
                            SELECT 
                        COL_NAME(fc.parent_object_id, fc.parent_column_id) as column_name,
                        OBJECT_SCHEMA_NAME(f.referenced_object_id) as referenced_schema,
                        OBJECT_NAME(f.referenced_object_id) as referenced_table,
                        COL_NAME(fc.referenced_object_id, fc.referenced_column_id) as referenced_column
                    FROM sys.foreign_keys AS f
                    INNER JOIN sys.foreign_key_columns AS fc 
                        ON f.object_id = fc.constraint_object_id
                    WHERE OBJECT_SCHEMA_NAME(f.parent_object_id) = ?
                    AND OBJECT_NAME(f.parent_object_id) = ?
                    """
                    cursor.execute(fk_query, (schema_name, table_name))
                    
                    for fk_row in cursor.fetchall():
                        for col in columns:
                            if col["name"] == fk_row[0]:
                                col["is_foreign_key"] = "YES"
                                col["foreign_key_reference"] = {
                                    "schema": fk_row[1],
                                    "table": fk_row[2],
                                    "column": fk_row[3]
                                }
                                break
        except Exception as e:
                    self.logger.warning(f"Error getting foreign key info: {str(e)}")
                
                # Get primary key information
                try:
                    pk_query = """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
                    AND TABLE_SCHEMA = ?
                    AND TABLE_NAME = ?
                    """
                    cursor.execute(pk_query, (schema_name, table_name))
                    
                    for pk_row in cursor.fetchall():
                        pk_column = pk_row[0]
                        for col in columns:
                            if col["name"] == pk_column:
                                col["is_primary_key"] = "YES"
                                break
        except Exception as e:
                    self.logger.warning(f"Error getting primary key info: {str(e)}")
                
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
                AND t.TABLE_NAME NOT IN ({})
                GROUP BY t.TABLE_SCHEMA, t.TABLE_NAME
                ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME
                """.format(','.join(['?' for _ in self.system_tables]))
                cursor.execute(query, list(self.system_tables))
                
                tables = []
                for row in cursor.fetchall():
                    columns = row[3].split(", ") if row[3] else []
                    tables.append({
                        "schema": row[0],
                        "name": row[1],
                        "column_count": row[2],
                        "columns": columns
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
    
    async def _get_hierarchical_data(self, parent_table: Dict[str, Any], child_table: Dict[str, Any], 
                                   relationship_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get hierarchical data from any parent-child table relationship.
        
        Args:
            parent_table: Dict containing parent table details:
                - schema: Schema name
                - name: Table name
                - key_column: Primary key column
                - display_column: Column to use for display name
                - additional_columns: Optional list of additional columns to fetch
            child_table: Dict containing child table details:
                - schema: Schema name
                - name: Table name
                - key_column: Primary key column
                - display_column: Column to use for display name
                - parent_key_column: Foreign key column referencing parent
                - additional_columns: Optional list of additional columns to fetch
            relationship_config: Dict containing relationship configuration:
                - type: Type of relationship (e.g., "one_to_many", "many_to_many")
                - junction_table: Optional junction table details for many-to-many
                - custom_conditions: Optional additional JOIN conditions
        
        Returns:
            Dict containing success status and hierarchical data structure
        """
        try:
            if not parent_table or not child_table:
                raise ValueError("Both parent and child table configurations are required")

            # Validate required fields
            required_fields = ["schema", "name", "key_column", "display_column"]
            for field in required_fields:
                if field not in parent_table or field not in child_table:
                    raise ValueError(f"Missing required field '{field}' in table configuration")
            
            if "parent_key_column" not in child_table:
                raise ValueError("Missing parent_key_column in child table configuration")

            # Build column lists
            parent_columns = [
                f"P.{parent_table['key_column']}", 
                f"P.{parent_table['display_column']}"
            ]
            child_columns = [
                f"C.{child_table['key_column']}", 
                f"C.{child_table['display_column']}"
            ]

            # Add additional columns if specified
            if parent_table.get("additional_columns"):
                parent_columns.extend([f"P.{col}" for col in parent_table["additional_columns"]])
            if child_table.get("additional_columns"):
                child_columns.extend([f"C.{col}" for col in child_table["additional_columns"]])

            # Build the query
            query = f"""
            SELECT 
                {', '.join(parent_columns)},
                {', '.join(child_columns)}
            FROM {parent_table['schema']}.{parent_table['name']} P
            LEFT JOIN {child_table['schema']}.{child_table['name']} C 
                ON P.{parent_table['key_column']} = C.{child_table['parent_key_column']}
            """

            # Add custom conditions if specified
            if relationship_config and relationship_config.get("custom_conditions"):
                query += f" AND {relationship_config['custom_conditions']}"

            query += f" ORDER BY P.{parent_table['display_column']}, C.{child_table['display_column']}"
            
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Process results into hierarchical structure
            hierarchy = {}
            for row in cursor.fetchall():
                parent_key = row[0]
                parent_name = row[1]
                child_key = row[2]
                child_name = row[3]
                
                # Get additional columns if any
                parent_extra = {}
                child_extra = {}
                col_index = 4
                
                if parent_table.get("additional_columns"):
                    for col in parent_table["additional_columns"]:
                        parent_extra[col] = row[col_index]
                        col_index += 1
                
                if child_table.get("additional_columns"):
                    for col in child_table["additional_columns"]:
                        child_extra[col] = row[col_index]
                        col_index += 1
                
                if parent_key not in hierarchy:
                    hierarchy[parent_key] = {
                        "id": parent_key,
                        "name": parent_name,
                        "children": [],
                        **parent_extra
                    }
                
                if child_key and child_name:
                    child_data = {
                        "id": child_key,
                        "name": child_name,
                        **child_extra
                    }
                    if child_data not in hierarchy[parent_key]["children"]:
                        hierarchy[parent_key]["children"].append(child_data)
                
                return {
                    "success": True,
                "data": list(hierarchy.values())
                }
                
        except Exception as e:
            self.logger.error(f"Error getting hierarchical data: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _get_join_patterns(self, table_name: str, schema_name: str = "dbo") -> Dict[str, Any]:
        """Get join patterns for a specific table."""
        try:
            if not table_name:
                raise ValueError("Table name is required")

            # Get foreign key relationships
            cursor = self.connection.cursor()
            
            # Query to get foreign keys where this table is the source
            outgoing_fk_query = """
        SELECT 
                fk.name AS constraint_name,
                OBJECT_SCHEMA_NAME(fk.parent_object_id) AS source_schema,
                OBJECT_NAME(fk.parent_object_id) AS source_table,
                c1.name AS source_column,
                OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS target_schema,
                OBJECT_NAME(fk.referenced_object_id) AS target_table,
                c2.name AS target_column
            FROM sys.foreign_keys fk
            INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            INNER JOIN sys.columns c1 ON fkc.parent_column_id = c1.column_id AND fkc.parent_object_id = c1.object_id
            INNER JOIN sys.columns c2 ON fkc.referenced_column_id = c2.column_id AND fkc.referenced_object_id = c2.object_id
            WHERE OBJECT_SCHEMA_NAME(fk.parent_object_id) = ?
            AND OBJECT_NAME(fk.parent_object_id) = ?
            """
            
            # Query to get foreign keys where this table is the target
            incoming_fk_query = """
        SELECT 
                fk.name AS constraint_name,
                OBJECT_SCHEMA_NAME(fk.parent_object_id) AS source_schema,
                OBJECT_NAME(fk.parent_object_id) AS source_table,
                c1.name AS source_column,
                OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS target_schema,
                OBJECT_NAME(fk.referenced_object_id) AS target_table,
                c2.name AS target_column
            FROM sys.foreign_keys fk
            INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            INNER JOIN sys.columns c1 ON fkc.parent_column_id = c1.column_id AND fkc.parent_object_id = c1.object_id
            INNER JOIN sys.columns c2 ON fkc.referenced_column_id = c2.column_id AND fkc.referenced_object_id = c2.object_id
            WHERE OBJECT_SCHEMA_NAME(fk.referenced_object_id) = ?
            AND OBJECT_NAME(fk.referenced_object_id) = ?
            """
            
            # Get outgoing relationships
            cursor.execute(outgoing_fk_query, (schema_name, table_name))
            outgoing_relationships = cursor.fetchall()
            
            # Get incoming relationships
            cursor.execute(incoming_fk_query, (schema_name, table_name))
            incoming_relationships = cursor.fetchall()
            
            # Process relationships into join patterns
            join_patterns = []
            
            # Pattern 1: Direct relationships (outgoing)
            if outgoing_relationships:
                pattern = {
                    "description": f"Direct relationships from {table_name}",
                    "related_tables": []
                }
                
                for rel in outgoing_relationships:
                    pattern["related_tables"].append({
                        "table": f"{rel[4]}.{rel[5]}",  # target_schema.target_table
                        "join_type": "INNER JOIN",
                        "source_column": rel[3],  # source_column
                        "target_column": rel[6],  # target_column
                        "constraint_name": rel[0]  # constraint_name
                    })
                
                join_patterns.append(pattern)
            
            # Pattern 2: Direct relationships (incoming)
            if incoming_relationships:
                pattern = {
                    "description": f"Direct relationships to {table_name}",
                    "related_tables": []
                }
                
                for rel in incoming_relationships:
                    pattern["related_tables"].append({
                        "table": f"{rel[1]}.{rel[2]}",  # source_schema.source_table
                        "join_type": "INNER JOIN",
                        "source_column": rel[6],  # target_column (since we're the target)
                        "target_column": rel[3],  # source_column (since we're the target)
                        "constraint_name": rel[0]  # constraint_name
                    })
                
                join_patterns.append(pattern)
                    
                    return {
                        "success": True,
                "data": {
                    "table_name": table_name,
                    "schema_name": schema_name,
                    "join_patterns": join_patterns
                }
                    }
                
        except Exception as e:
            self.logger.error(f"Error getting join patterns for table {table_name}: {str(e)}")
            return {"success": False, "error": str(e)} 