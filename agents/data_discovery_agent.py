from typing import Dict, List, Optional
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy import inspect
import logging
from pathlib import Path
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class DataDiscoveryAgent(BaseAgent):
    def __init__(self, config: Dict):
        """
        Initialize the Data Discovery Agent.
        
        Args:
            config: Configuration dictionary containing database settings
        """
        super().__init__(config)
        self.engine = None
        self.inspector = None
        
    async def initialize(self) -> bool:
        """Initialize the agent and establish database connection."""
        try:
            connection_string = self._build_connection_string()
            self.engine = self._create_engine(connection_string)
            self.inspector = inspect(self.engine)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DataDiscoveryAgent: {str(e)}")
            return False
        
    def _create_engine(self, connection_string: str) -> Engine:
        """Create SQLAlchemy engine with proper configuration."""
        try:
            return sa.create_engine(connection_string)
        except Exception as e:
            logger.error(f"Failed to create database engine: {str(e)}")
            raise
            
    def _build_connection_string(self) -> str:
        """Build connection string from config."""
        db_config = self.config.get("database", {})
        driver = db_config.get("driver", "mysql")
        server = db_config.get("server", "localhost")
        database = db_config.get("database")
        username = db_config.get("username")
        password = db_config.get("password")
        
        if not all([database, username, password]):
            raise ValueError("Missing required database configuration")
            
        return f"{driver}://{username}:{password}@{server}/{database}"
            
    async def process(self, data: Dict) -> Dict:
        """
        Process incoming requests.
        
        Args:
            data: Dictionary containing the request parameters
            
        Returns:
            Dictionary containing the processing results
        """
        action = data.get("action")
        if action == "get_schema":
            return {"result": self.get_schema_info()}
        elif action == "analyze_relationships":
            return {"result": self.analyze_table_relationships()}
        elif action == "suggest_joins":
            source = data.get("source_table")
            target = data.get("target_table")
            if not (source and target):
                raise ValueError("Missing source or target table")
            return {"result": self.suggest_joins(source, target)}
        elif action == "validate_query":
            query = data.get("query")
            if not query:
                raise ValueError("Missing query")
            return {"result": self.validate_query(query)}
        else:
            raise ValueError(f"Unknown action: {action}")
            
    async def cleanup(self):
        """Cleanup resources."""
        if self.engine:
            self.engine.dispose()
            
    def get_schema_info(self) -> Dict:
        """
        Get comprehensive schema information for the database.
        
        Returns:
            Dict containing tables, columns, and relationships
        """
        try:
            schema_info = {
                "tables": {},
                "relationships": []
            }
            
            # Get all tables
            for table_name in self.inspector.get_table_names():
                columns = []
                for column in self.inspector.get_columns(table_name):
                    columns.append({
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column["nullable"]
                    })
                
                # Get primary key info
                pk_constraint = self.inspector.get_pk_constraint(table_name)
                primary_keys = pk_constraint["constrained_columns"] if pk_constraint else []
                
                # Get foreign key info
                foreign_keys = []
                for fk in self.inspector.get_foreign_keys(table_name):
                    foreign_keys.append({
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"],
                        "constrained_columns": fk["constrained_columns"]
                    })
                    # Add to relationships list
                    schema_info["relationships"].append({
                        "source_table": table_name,
                        "target_table": fk["referred_table"],
                        "source_columns": fk["constrained_columns"],
                        "target_columns": fk["referred_columns"]
                    })
                
                schema_info["tables"][table_name] = {
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys
                }
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting schema info: {str(e)}")
            raise
            
    def analyze_table_relationships(self) -> List[Dict]:
        """
        Analyze and detect relationships between tables.
        
        Returns:
            List of detected relationships with confidence scores
        """
        relationships = []
        try:
            schema_info = self.get_schema_info()
            
            # Analyze explicit relationships (foreign keys)
            for rel in schema_info["relationships"]:
                relationships.append({
                    "source_table": rel["source_table"],
                    "target_table": rel["target_table"],
                    "type": "foreign_key",
                    "confidence": 1.0,
                    "columns": {
                        "source": rel["source_columns"],
                        "target": rel["target_columns"]
                    }
                })
            
            # TODO: Implement implicit relationship detection
            # This could include:
            # 1. Column name matching
            # 2. Data type compatibility
            # 3. Value overlap analysis
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing relationships: {str(e)}")
            raise
            
    def suggest_joins(self, source_table: str, target_table: str) -> List[Dict]:
        """
        Suggest possible join paths between two tables.
        
        Args:
            source_table: Starting table name
            target_table: Target table name
            
        Returns:
            List of possible join paths with metadata
        """
        try:
            relationships = self.analyze_table_relationships()
            join_paths = []
            
            # Direct relationships
            direct_rels = [
                rel for rel in relationships
                if (rel["source_table"] == source_table and rel["target_table"] == target_table) or
                   (rel["source_table"] == target_table and rel["target_table"] == source_table)
            ]
            
            for rel in direct_rels:
                join_paths.append({
                    "path": [source_table, target_table],
                    "joins": [{
                        "type": rel["type"],
                        "conditions": [
                            f"{rel['source_table']}.{col_src} = {rel['target_table']}.{col_tgt}"
                            for col_src, col_tgt in zip(
                                rel["columns"]["source"],
                                rel["columns"]["target"]
                            )
                        ]
                    }],
                    "confidence": rel["confidence"]
                })
            
            # TODO: Implement indirect join path detection
            # This would find paths through intermediate tables
            
            return join_paths
            
        except Exception as e:
            logger.error(f"Error suggesting joins: {str(e)}")
            raise
            
    def validate_query(self, query: str) -> Dict:
        """
        Validate a SQL query for correctness and optimization opportunities.
        
        Args:
            query: SQL query string to validate
            
        Returns:
            Dict containing validation results and optimization suggestions
        """
        try:
            # Parse the query
            statement = sa.text(query)
            
            # Execute explain plan
            with self.engine.connect() as conn:
                explain_result = conn.execute(
                    f"EXPLAIN {query}"
                ).fetchall()
            
            # Basic validation passed if we get here
            result = {
                "is_valid": True,
                "explain_plan": explain_result,
                "suggestions": []
            }
            
            # TODO: Add query optimization suggestions based on explain plan analysis
            
            return result
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "suggestions": []
            } 