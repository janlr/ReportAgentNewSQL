from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import json
import logging
from pathlib import Path
import hashlib
from datetime import datetime
import inspect
import ast
import re
from dataclasses import dataclass
from .base_agent import BaseAgent

@dataclass
class ReportTemplate:
    """Data class for report template metadata."""
    id: str
    name: str
    description: str
    category: str
    tags: List[str]
    parameters: Dict[str, Any]
    version: int = 1
    created_at: str = None
    updated_at: str = None
    usage_count: int = 0
    previous_versions: List[str] = None
    client_name: str = "default"  # Added client name field
    joins: List[Dict[str, str]] = None  # Added joins field
    group_by: List[str] = None  # Added group by field
    having: Dict[str, str] = None  # Added having conditions
    order_by: List[Dict[str, str]] = None  # Added order by field

class ReportGeneratorAgent(BaseAgent):
    """Agent responsible for managing report templates and generation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        super().__init__("report_generator_agent")
        self.config = config
        self.client_configs = config.get("client_configs", {})
        self.current_client = config.get("default_client", "default")
        
        # Set up directories
        self.base_dir = Path(config.get("reports_dir", "./reports"))
        self.templates_dir = self.base_dir / "templates"
        self.versions_dir = self.base_dir / "versions"
        self.output_dir = self.base_dir / "output"
        self.client_config_dir = self.base_dir / "client_configs"
        
        for directory in [self.base_dir, self.templates_dir, self.versions_dir, 
                         self.output_dir, self.client_config_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
        self.load_client_configs()
    
    def load_client_configs(self) -> None:
        """Load client configurations from files."""
        try:
            # Load built-in configs
            self.client_configs.update(self.config.get("CLIENT_CONFIGS", {}))
            
            # Load custom client configs from files
            for config_file in self.client_config_dir.glob("*.json"):
                with open(config_file, 'r') as f:
                    client_name = config_file.stem
                    self.client_configs[client_name] = json.load(f)
                    
            self.logger.info(f"Loaded configurations for clients: {list(self.client_configs.keys())}")
        except Exception as e:
            self.logger.error(f"Error loading client configs: {str(e)}")
    
    def validate_client_config(self, config: Dict[str, Any]) -> bool:
        """Validate client configuration structure."""
        required_keys = ["report_categories", "metrics_mapping", "table_mapping", "field_mapping"]
        if not all(key in config for key in required_keys):
            return False
            
        # Validate metrics mapping structure
        if not isinstance(config["metrics_mapping"], dict):
            return False
        
        # Validate table mapping
        if not isinstance(config["table_mapping"], dict):
            return False
            
        # Validate field mapping
        if not isinstance(config["field_mapping"], dict):
            return False
            
        return True
    
    def add_client_config(self, client_name: str, config: Dict[str, Any]) -> bool:
        """Add a new client configuration."""
        try:
            if not self.validate_client_config(config):
                raise ValueError("Invalid client configuration structure")
                
            # Save to file
            config_path = self.client_config_dir / f"{client_name}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            # Update in memory
            self.client_configs[client_name] = config
            self.logger.info(f"Added configuration for client: {client_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding client config: {str(e)}")
            return False
    
    def get_client_metrics(self, category: str) -> Dict[str, str]:
        """Get metrics mapping for current client and category."""
        client_config = self.client_configs.get(self.current_client, {})
        return client_config.get("metrics_mapping", {}).get(category, {})
    
    def get_table_mapping(self, table_name: str) -> str:
        """Get actual table name for current client."""
        client_config = self.client_configs.get(self.current_client, {})
        return client_config.get("table_mapping", {}).get(table_name, table_name)
    
    def get_field_mapping(self, field_name: str) -> str:
        """Get actual field name for current client."""
        client_config = self.client_configs.get(self.current_client, {})
        return client_config.get("field_mapping", {}).get(field_name, field_name)
    
    def generate_query(self, template: ReportTemplate) -> str:
        """Generate SQL query based on template and client configuration."""
        try:
            client_config = self.client_configs.get(template.client_name, 
                                                  self.client_configs["default"])
            
            # Get metrics for the template category
            metrics = client_config["metrics_mapping"].get(template.category, {})
            
            # Build SELECT clause
            select_clause = []
            for metric_name, metric_expr in metrics.items():
                if metric_name in template.parameters.get("metrics", []):
                    select_clause.append(f"{metric_expr} as {metric_name}")
            
            # Add dimensions to SELECT clause
            dimensions = template.parameters.get("dimensions", [])
            for dim in dimensions:
                mapped_field = self.get_field_mapping(dim)
                select_clause.append(f"{mapped_field} as {dim}")
            
            # Build FROM clause with proper table names and JOINs
            from_table = self.get_table_mapping(template.parameters.get("main_table", ""))
            join_clauses = []
            
            if template.joins:
                for join in template.joins:
                    join_type = join.get("type", "INNER")
                    from_table_field = self.get_field_mapping(join["from_field"])
                    join_table = self.get_table_mapping(join["table"])
                    join_field = self.get_field_mapping(join["join_field"])
                    join_clauses.append(
                        f"{join_type} JOIN {join_table} ON "
                        f"{from_table}.{from_table_field} = {join_table}.{join_field}"
                    )
            
            # Build WHERE clause with mapped field names
            where_conditions = []
            for field, value in template.parameters.get("filters", {}).items():
                mapped_field = self.get_field_mapping(field)
                if isinstance(value, (int, float)):
                    where_conditions.append(f"{mapped_field} = {value}")
                else:
                    where_conditions.append(f"{mapped_field} = '{value}'")
            
            # Build GROUP BY clause
            group_by_clause = ""
            if template.group_by:
                group_fields = [self.get_field_mapping(f) for f in template.group_by]
                group_by_clause = f"GROUP BY {', '.join(group_fields)}"
            
            # Build HAVING clause
            having_clause = ""
            if template.having:
                having_conditions = []
                for field, condition in template.having.items():
                    mapped_field = self.get_field_mapping(field)
                    having_conditions.append(f"{mapped_field} {condition}")
                having_clause = f"HAVING {' AND '.join(having_conditions)}"
            
            # Build ORDER BY clause
            order_by_clause = ""
            if template.order_by:
                order_terms = []
                for order in template.order_by:
                    field = self.get_field_mapping(order["field"])
                    direction = order.get("direction", "ASC")
                    order_terms.append(f"{field} {direction}")
                order_by_clause = f"ORDER BY {', '.join(order_terms)}"
            
            # Construct final query
            query_parts = [
                f"SELECT {', '.join(select_clause)}",
                f"FROM {from_table}"
            ]
            
            if join_clauses:
                query_parts.extend(join_clauses)
            
            if where_conditions:
                query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
            
            if group_by_clause:
                query_parts.append(group_by_clause)
            
            if having_clause:
                query_parts.append(having_clause)
            
            if order_by_clause:
                query_parts.append(order_by_clause)
            
            return " ".join(query_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating query: {str(e)}")
            return ""
    
    def export_client_config(self, client_name: str, output_path: Optional[Path] = None) -> bool:
        """Export client configuration to a file."""
        try:
            if client_name not in self.client_configs:
                raise ValueError(f"Client configuration not found: {client_name}")
            
            config = self.client_configs[client_name]
            
            if output_path is None:
                output_path = self.client_config_dir / f"{client_name}_export.json"
            
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            self.logger.info(f"Exported configuration for client {client_name} to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting client config: {str(e)}")
            return False
    
    def import_client_config(self, config_path: Union[str, Path], client_name: Optional[str] = None) -> bool:
        """Import client configuration from a file."""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise ValueError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if not self.validate_client_config(config):
                raise ValueError("Invalid client configuration structure")
            
            if client_name is None:
                client_name = config_path.stem
            
            return self.add_client_config(client_name, config)
            
        except Exception as e:
            self.logger.error(f"Error importing client config: {str(e)}")
            return False
    
    def parse_user_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse natural language prompt into report parameters."""
        try:
            # Extract time-related phrases
            time_patterns = {
                'last_n_days': r'last (\d+) days',
                'last_n_months': r'last (\d+) months',
                'date_range': r'between (\d{4}-\d{2}-\d{2}) and (\d{4}-\d{2}-\d{2})',
                'year': r'(in|for) (\d{4})',
                'month': r'(in|for) (January|February|March|April|May|June|July|August|September|October|November|December)'
            }
            
            time_filters = {}
            for key, pattern in time_patterns.items():
                matches = re.findall(pattern, prompt.lower())
                if matches:
                    time_filters[key] = matches[0]
            
            # Extract metrics
            metric_patterns = {
                'sales': r'(total |)(sales|revenue)',
                'orders': r'(number of |total |)(orders|transactions)',
                'customers': r'(unique |distinct |)(customers|clients)',
                'average': r'average (order value|purchase|sale)',
                'growth': r'(growth|increase|decrease)',
                'comparison': r'compare|versus|vs'
            }
            
            metrics = []
            for key, pattern in metric_patterns.items():
                if re.search(pattern, prompt.lower()):
                    metrics.append(key)
            
            # Extract dimensions
            dimension_patterns = {
                'product': r'by (product|item)',
                'category': r'by (category|type)',
                'region': r'by (region|location|country)',
                'customer': r'by (customer|client)',
                'time': r'by (month|quarter|year)'
            }
            
            dimensions = []
            for key, pattern in dimension_patterns.items():
                if re.search(pattern, prompt.lower()):
                    dimensions.append(key)
            
            # Extract sorting preferences
            sort_patterns = {
                'top': r'top (\d+)',
                'bottom': r'bottom (\d+)',
                'sort': r'sort by|order by'
            }
            
            sorting = {}
            for key, pattern in sort_patterns.items():
                matches = re.findall(pattern, prompt.lower())
                if matches:
                    sorting[key] = matches[0]
            
            return {
                'time_filters': time_filters,
                'metrics': metrics,
                'dimensions': dimensions,
                'sorting': sorting,
                'original_prompt': prompt
            }
        except Exception as e:
            self.logger.error(f"Error parsing prompt: {str(e)}")
            return {}

    def optimize_query(self, query: str) -> Tuple[str, List[str]]:
        """Optimize the generated SQL query."""
        try:
            optimizations = []
            
            # Add index hints for large tables
            if 'sales' in query.lower() or 'orders' in query.lower():
                query = re.sub(
                    r'FROM\s+([^\s]+)',
                    r'FROM \1 WITH (INDEX(IX_OrderDate))',
                    query
                )
                optimizations.append("Added index hint for OrderDate")
            
            # Add OPTION (RECOMPILE) for complex queries
            if query.count('JOIN') > 2 or 'GROUP BY' in query:
                query += " OPTION (RECOMPILE)"
                optimizations.append("Added RECOMPILE hint for complex query")
            
            # Add TOP clause for large result sets
            if 'ORDER BY' in query and 'TOP' not in query:
                query = re.sub(
                    r'SELECT\s+',
                    'SELECT TOP 10000 ',
                    query
                )
                optimizations.append("Added TOP clause to limit result set")
            
            # Add query hints for parallel execution
            if query.count('JOIN') > 1:
                query += " OPTION (MAXDOP 4)"
                optimizations.append("Added parallel execution hint")
            
            return query, optimizations
            
        except Exception as e:
            self.logger.error(f"Error optimizing query: {str(e)}")
            return query, ["Query optimization failed"]

    def create_example_configs(self) -> None:
        """Create example client configurations."""
        try:
            # E-commerce client example
            ecommerce_config = {
                "report_categories": ["sales", "inventory", "customer"],
                "metrics_mapping": {
                    "sales": {
                        "total_revenue": "SUM(OrderDetails.UnitPrice * OrderDetails.Quantity)",
                        "order_count": "COUNT(DISTINCT Orders.OrderID)",
                        "average_order_value": "AVG(OrderDetails.UnitPrice * OrderDetails.Quantity)",
                        "growth_rate": """
                            ((SUM(CASE WHEN YEAR(OrderDate) = YEAR(GETDATE())
                              THEN OrderDetails.UnitPrice * OrderDetails.Quantity ELSE 0 END) /
                              NULLIF(SUM(CASE WHEN YEAR(OrderDate) = YEAR(GETDATE()) - 1
                              THEN OrderDetails.UnitPrice * OrderDetails.Quantity ELSE 0 END), 0) - 1) * 100
                        """
                    },
                    "inventory": {
                        "stock_level": "SUM(UnitsInStock)",
                        "reorder_count": "COUNT(CASE WHEN UnitsInStock <= ReorderLevel THEN 1 END)",
                        "inventory_value": "SUM(UnitsInStock * UnitPrice)"
                    },
                    "customer": {
                        "customer_count": "COUNT(DISTINCT CustomerID)",
                        "repeat_rate": """
                            COUNT(CASE WHEN OrderCount > 1 THEN 1 END) * 100.0 / 
                            COUNT(DISTINCT CustomerID)
                        """,
                        "average_lifetime_value": """
                            SUM(OrderDetails.UnitPrice * OrderDetails.Quantity) / 
                            COUNT(DISTINCT Orders.CustomerID)
                        """
                    }
                },
                "table_mapping": {
                    "orders": "Sales.Orders",
                    "order_details": "Sales.OrderDetails",
                    "products": "Production.Products",
                    "customers": "Sales.Customers",
                    "categories": "Production.Categories"
                },
                "field_mapping": {
                    "order_date": "OrderDate",
                    "product_name": "ProductName",
                    "category_name": "CategoryName",
                    "customer_id": "CustomerID",
                    "unit_price": "UnitPrice",
                    "quantity": "Quantity"
                }
            }
            
            # Manufacturing client example
            manufacturing_config = {
                "report_categories": ["production", "quality", "maintenance"],
                "metrics_mapping": {
                    "production": {
                        "output_quantity": "SUM(ProductionOutput.Quantity)",
                        "efficiency_rate": "AVG(ActualOutput / PlannedOutput) * 100",
                        "downtime_hours": "SUM(DATEDIFF(hour, StartTime, EndTime))"
                    },
                    "quality": {
                        "defect_rate": "COUNT(DefectiveItems) * 100.0 / COUNT(TotalItems)",
                        "first_pass_yield": "SUM(PassedFirstInspection) * 100.0 / COUNT(TotalItems)",
                        "scrap_cost": "SUM(ScrapQuantity * MaterialCost)"
                    },
                    "maintenance": {
                        "mtbf": "AVG(DATEDIFF(hour, LastFailure, NextFailure))",
                        "repair_cost": "SUM(MaintenanceCost)",
                        "preventive_maintenance_ratio": """
                            COUNT(CASE WHEN MaintenanceType = 'Preventive' THEN 1 END) * 100.0 /
                            COUNT(MaintenanceType)
                        """
                    }
                },
                "table_mapping": {
                    "production_output": "Production.Output",
                    "quality_control": "Quality.Inspections",
                    "maintenance_logs": "Maintenance.Logs",
                    "equipment": "Production.Equipment"
                },
                "field_mapping": {
                    "production_date": "ProductionDate",
                    "machine_id": "EquipmentID",
                    "product_code": "ProductCode",
                    "operator_id": "OperatorID",
                    "shift": "ShiftCode"
                }
            }
            
            # Save example configurations
            self.add_client_config("ecommerce_example", ecommerce_config)
            self.add_client_config("manufacturing_example", manufacturing_config)
            
        except Exception as e:
            self.logger.error(f"Error creating example configs: {str(e)}")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process report generation request."""
        try:
            action = data.get("action", "")
            
            if action == "generate_report":
                if "prompt" in data:
                    # Handle natural language prompt
                    parameters = self.parse_user_prompt(data["prompt"])
                    template = self.create_template_from_parameters(parameters)
                else:
                    # Handle direct template-based request
                    template_id = data.get("template_id")
                    template = self.load_template(template_id)
                
                if template:
                    query = self.generate_query(template)
                    optimized_query, optimizations = self.optimize_query(query)
                    return {
                        "status": "success",
                        "query": optimized_query,
                        "optimizations": optimizations,
                        "parameters": template.parameters
                    }
            
            elif action == "add_client_config":
                client_name = data.get("client_name")
                config = data.get("config")
                success = self.add_client_config(client_name, config)
                return {"status": "success" if success else "error"}
                
            elif action == "export_client_config":
                client_name = data.get("client_name")
                output_path = data.get("output_path")
                success = self.export_client_config(client_name, output_path)
                return {"status": "success" if success else "error"}
                
            elif action == "import_client_config":
                config_path = data.get("config_path")
                client_name = data.get("client_name")
                success = self.import_client_config(config_path, client_name)
                return {"status": "success" if success else "error"}
                
            return {"status": "error", "message": "Invalid action"}
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return {"status": "error", "message": str(e)} 