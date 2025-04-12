from typing import Dict, Any, List, Optional, Union
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from .base_agent import BaseAgent
from .database_agent import DatabaseAgent
from .data_manager_agent import DataManagerAgent
from .user_interface_agent import UserInterfaceAgent
from .report_generator_agent import ReportGeneratorAgent
from .insight_generator_agent import InsightGeneratorAgent
from .assistant_agent import AssistantAgent
from .visualization_agent import VisualizationAgent

class MasterOrchestratorAgent(BaseAgent):
    """Agent responsible for coordinating all other agents."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str, anthropic_api_key: str):
        """Initialize the orchestrator."""
        super().__init__("master_orchestrator_agent", config)
        self.config = config
        self.output_dir = Path(output_dir)
        self.anthropic_api_key = anthropic_api_key
        
        # Initialize agent instances
        self.agents = {
            "database": DatabaseAgent(config["database"]),
            "data_manager": DataManagerAgent(config["data_manager"]),
            "user_interface": UserInterfaceAgent(config["user_interface"]),
            "visualization": VisualizationAgent(config["visualization"]),
            "report_generator": ReportGeneratorAgent(
                report_config=config["report_generator"]["config"],
                db_config=config["database"],
                output_dir=config["report_generator"]["output_dir"],
                anthropic_api_key=self.anthropic_api_key,
                orchestrator=self
            ),
            "insight_generator": InsightGeneratorAgent(config["insight_generator"]),
            "assistant": AssistantAgent(config.get("assistant", {}))
        }
        self.workflow_history = []
    
    async def initialize(self) -> bool:
        """Initialize all agents."""
        try:
            self.logger.info("Initializing database agent...")
            if not await self.agents["database"].initialize():
                raise RuntimeError("Failed to initialize database agent")

            self.logger.info("Initializing visualization agent...")
            if not await self.agents["visualization"].initialize():
                raise RuntimeError("Failed to initialize visualization agent")

            self.logger.info("Initializing report generator...")
            if not await self.agents["report_generator"].initialize():
                raise RuntimeError("Failed to initialize report generator")

            # Make insight generator optional
            self.logger.info("Initializing insight generator...")
            try:
                if not await self.agents["insight_generator"].initialize():
                    self.logger.warning("Failed to initialize insight generator, but continuing anyway")
            except Exception as e:
                self.logger.warning(f"Failed to initialize insight generator: {str(e)}")
                self.logger.warning("Continuing without insight generation capabilities")

            self.logger.info("All required agents initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {str(e)}", exc_info=True)
            # Re-raise with more context
            raise RuntimeError(f"Master orchestrator initialization failed: {str(e)}") from e
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process workflow requests."""
        try:
            workflow = input_data.get("workflow")
            if not workflow:
                raise ValueError("Workflow not specified")

            result = await getattr(self, f"_handle_{workflow}")(input_data)
            return result
        except Exception as e:
            return await self._handle_agent_error("master_orchestrator", e)
    
    async def cleanup(self) -> bool:
        """Clean up all agents."""
        try:
            self.logger.info("Cleaning up master orchestrator")
            
            # Clean up each agent
            for name, agent in self.agents.items():
                try:
                    await agent.cleanup()
                    self.logger.info(f"Cleaned up {name} agent")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {name} agent: {str(e)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up master orchestrator: {str(e)}")
            return False
    
    async def _handle_report_generation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle report generation workflow by delegating to the ReportGeneratorAgent."""
        parameters = input_data.get("parameters", {})
        action = input_data.get("action")
        report_type_for_logging = parameters.get("report_type") or parameters.get("prompt", "N/A")

        # Log the start
        start_time = datetime.now()
        self.logger.info(f"Starting report generation workflow. Action: {action}, Details: {report_type_for_logging}")

        try:
            # Directly call the ReportGeneratorAgent, passing the original request.
            # It should handle data gathering, cleaning (if needed), generation, insights, etc., based on the action.
            report_result = await self.agents["report_generator"].process(input_data)

            # Check if the report generation itself failed
            if not report_result.get("success"):
                 # Log the specific error from the agent
                 agent_error = report_result.get('error', 'Unknown error from ReportGeneratorAgent')
                 self.logger.error(f"ReportGeneratorAgent failed during processing: {agent_error}")
                 raise RuntimeError(f"ReportGeneratorAgent failed: {agent_error}")

            # Log completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.logger.info(f"Report generation workflow completed successfully in {duration:.2f}s.")
            self.workflow_history.append({
                "workflow": "report_generation",
                "action": action,
                "parameters": parameters,
                "timestamp": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "completed"
            })
            return report_result # Return the successful result from ReportGeneratorAgent

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            # Log the error with traceback
            self.logger.error(f"Error in report generation workflow '{action}' after {duration:.2f}s: {str(e)}", exc_info=True)

            # Log failure details
            self.workflow_history.append({
                "workflow": "report_generation",
                "action": action,
                "parameters": parameters,
                "timestamp": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "failed",
                "error": str(e)
            })
            # Return a standardized error format including the original error type/message
            return {"success": False, "error": f"Master orchestrator workflow failed: {type(e).__name__}: {str(e)}"}
    
    async def _handle_data_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data analysis workflow."""
        try:
            action = input_data.get("action")
            parameters = input_data.get("parameters", {})
            
            if action == "get_schema_info":
                # Pass table_name and schema_name parameters if provided
                return await self.agents["database"].process({
                    "action": "get_schema_info",
                    "table_name": parameters.get("table_name"),
                    "schema_name": parameters.get("schema_name", "dbo")
                })
            
            elif action == "get_product_categories":
                return await self.agents["database"].process({
                    "action": "get_product_categories"
                })
            
            elif action == "analyze_sales":
                return await self.agents["data_manager"].process({
                    "action": "analyze_sales",
                    "parameters": parameters
                })
            
            elif action == "get_sample_data":
                return await self.agents["database"].process({
                    "action": "get_sample_data",
                    "parameters": parameters
                })
            
            elif action == "analyze_schema":
                # Get schema information from database
                schema_result = await self.agents["database"].process({
                    "action": "get_schema_info"
                })
                
                return schema_result
            
            elif action == "analyze_relationships":
                # Analyze table relationships
                relationships_result = await self.agents["database"].process({
                    "action": "analyze_relationships"
                })
                
                return relationships_result
            
            elif action == "suggest_joins":
                # Get join suggestions
                joins_result = await self.agents["database"].process({
                    "action": "suggest_joins",
                    "source_table": parameters.get("source_table"),
                    "target_table": parameters.get("target_table")
                })
                
                return joins_result
            
            elif action == "execute_query":
                # Execute a query
                query_params = {
                    "action": "execute_query",
                    "query": parameters.get("query")
                }
                
                # Only add params if they exist
                if parameters.get("params"):
                    query_params["params"] = parameters.get("params")
                
                return await self.agents["database"].process(query_params)
            
            else:
                raise ValueError(f"Unknown data analysis action: {action}")
            
        except Exception as e:
            self.logger.error(f"Error in data analysis workflow: {str(e)}")
            return await self._handle_agent_error("data_analysis", e)
    
    async def _handle_database_management(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database management workflow."""
        try:
            action = input_data.get("action")
            parameters = input_data.get("parameters", {})
            
            if action == "validate_query":
                # Validate SQL query
                validation_result = await self.agents["database"].process({
                    "action": "validate_query",
                    "query": parameters.get("query")
                })
                
                return validation_result
            
            else:
                raise ValueError(f"Unknown database management action: {action}")
            
        except Exception as e:
            self.logger.error(f"Error in database management workflow: {str(e)}")
            raise

    async def _handle_data_visualization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data visualization workflow."""
        try:
            action = input_data.get("action")
            parameters = input_data.get("parameters", {})
            
            if action == "create_visualization":
                # Get data from database based on parameters
                query_params = {
                    "start_date": parameters.get("start_date"),
                    "end_date": parameters.get("end_date")
                }
                
                # Build query based on visualization type
                viz_type = parameters.get("viz_type")
                options = parameters.get("options", {})
                
                if viz_type == "Sales Trends":
                    query = """
                        SELECT 
                            CAST(SOH.OrderDate AS DATE) as OrderDate,
                            SUM(SOD.LineTotal) as TotalSales,
                            COUNT(DISTINCT SOH.SalesOrderID) as OrderCount
                        FROM Sales.SalesOrderHeader SOH
                        JOIN Sales.SalesOrderDetail SOD ON SOH.SalesOrderID = SOD.SalesOrderID
                        WHERE SOH.OrderDate BETWEEN @start_date AND @end_date
                        GROUP BY CAST(SOH.OrderDate AS DATE)
                        ORDER BY OrderDate
                    """
                
                elif viz_type == "Product Performance":
                    query = """
                        SELECT TOP (@top_n)
                            P.Name as ProductName,
                            SUM(SOD.LineTotal) as Revenue,
                            SUM(SOD.OrderQty) as UnitsSold,
                            AVG(SOD.UnitPrice) as AvgPrice
                        FROM Sales.SalesOrderHeader SOH
                        JOIN Sales.SalesOrderDetail SOD ON SOH.SalesOrderID = SOD.SalesOrderID
                        JOIN Production.Product P ON SOD.ProductID = P.ProductID
                        WHERE SOH.OrderDate BETWEEN @start_date AND @end_date
                        GROUP BY P.ProductID, P.Name
                        ORDER BY 
                            CASE WHEN @metric = 'Revenue' THEN SUM(SOD.LineTotal)
                                 WHEN @metric = 'Units Sold' THEN SUM(SOD.OrderQty)
                                 ELSE SUM(SOD.LineTotal) END DESC
                    """
                    query_params["top_n"] = options.get("top_n", 10)
                    query_params["metric"] = options.get("metric", "Revenue")
                
                elif viz_type == "Customer Segments":
                    query = """
                        SELECT 
                            C.CustomerID,
                            COUNT(DISTINCT SOH.SalesOrderID) as OrderCount,
                            AVG(SOH.TotalDue) as AvgOrderValue,
                            SUM(SOH.TotalDue) as TotalSpent
                        FROM Sales.Customer C
                        JOIN Sales.SalesOrderHeader SOH ON C.CustomerID = SOH.CustomerID
                        WHERE SOH.OrderDate BETWEEN @start_date AND @end_date
                        GROUP BY C.CustomerID
                    """
                
                elif viz_type == "Geographic Analysis":
                    query = """
                        SELECT 
                            ST.Name as Territory,
                            ST.CountryRegionCode as Region,
                            SUM(SOH.TotalDue) as TotalSales,
                            COUNT(DISTINCT SOH.CustomerID) as CustomerCount,
                            COUNT(DISTINCT SOH.SalesOrderID) as OrderCount
                        FROM Sales.SalesTerritory ST
                        JOIN Sales.SalesOrderHeader SOH ON ST.TerritoryID = SOH.TerritoryID
                        WHERE SOH.OrderDate BETWEEN @start_date AND @end_date
                        GROUP BY ST.TerritoryID, ST.Name, ST.CountryRegionCode
                    """
                
                else:
                    raise ValueError(f"Unsupported visualization type: {viz_type}")
                
                # Get data from database
                result = await self.agents["database"].process({
                    "action": "execute_query",
                    "query": query,
                    "params": query_params
                })
                
                if not result.get("success"):
                    return result
                
                # Clean data
                clean_result = await self.agents["data_manager"].process({
                    "action": "clean_data",
                    "data": result["data"]
                })
                
                if not clean_result.get("success"):
                    return clean_result
                
                # Generate visualization
                viz_result = await self.agents["report_generator"].process({
                    "action": "create_visualization",
                    "parameters": {
                        "data": clean_result["data"],
                        "viz_type": viz_type,
                        "options": options
                    }
                })
                
                # Generate insights if requested
                insights = None
                if options.get("include_insights", True):
                    insight_result = await self.agents["insight_generator"].process({
                        "action": "generate_insights",
                        "data": clean_result["data"],
                        "metadata": {
                            "viz_type": viz_type,
                            "parameters": parameters
                        }
                    })
                    if insight_result.get("success"):
                        insights = insight_result["data"]
                
                return {
                    "success": True,
                    "data": {
                        "chart": viz_result["data"]["chart"],
                        "insights": insights
                    }
                }
            
            else:
                raise ValueError(f"Unknown visualization action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error in data visualization workflow: {str(e)}")
            return await self._handle_agent_error("data_visualization", e)

    async def _handle_schema_configuration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle schema configuration workflow."""
        try:
            action = input_data.get("action")
            parameters = input_data.get("parameters", {})
            
            if action == "get_join_patterns":
                # Forward the request to the database agent
                return await self.agents["database"].process({
                    "action": "get_join_patterns",
                    "parameters": parameters
                })
            elif action == "configure_schema":
                # Get schema information from database
                schema_result = await self.agents["database"].process({
                    "action": "get_schema_info"
                })
                
                if not schema_result.get("success"):
                    return schema_result
                
                schema_info = schema_result.get("data", {})
                
                # Analyze tables and suggest categories
                table_categories = {
                    "fact_tables": [],
                    "dimension_tables": [],
                    "lookup_tables": [],
                    "transaction_tables": []
                }
                
                # Simple categorization based on naming conventions and structure
                for table in schema_info.get("tables", []):
                    table_name = table["name"].lower()
                    if "fact" in table_name or "sales" in table_name or "orders" in table_name:
                        table_categories["fact_tables"].append(f"{table['schema']}.{table['name']}")
                    elif "dim" in table_name or "dimension" in table_name:
                        table_categories["dimension_tables"].append(f"{table['schema']}.{table['name']}")
                    elif table.get("column_count", 0) <= 5:  # Simple heuristic for lookup tables
                        table_categories["lookup_tables"].append(f"{table['schema']}.{table['name']}")
                    elif "transaction" in table_name or "history" in table_name:
                        table_categories["transaction_tables"].append(f"{table['schema']}.{table['name']}")
                
                # Get join patterns for fact tables
                join_patterns = []
                for fact_table in table_categories["fact_tables"]:
                    schema_name, table_name = fact_table.split(".")
                    join_result = await self.agents["database"].process({
                        "action": "get_join_patterns",
                        "parameters": {
                            "table_name": table_name,
                            "schema_name": schema_name
                        }
                    })
                    
                    if join_result.get("success"):
                        join_patterns.extend(join_result["data"].get("join_patterns", []))
                
                return {
                    "success": True,
                    "data": {
                        "table_categories": table_categories,
                        "join_patterns": join_patterns
                    }
                }
            else:
                raise ValueError(f"Unknown action for schema configuration workflow: {action}")
                
        except Exception as e:
            self.logger.error(f"Error in schema configuration workflow: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _handle_agent_error(self, agent_name: str, error: Exception) -> Dict[str, Any]:
        """Handle agent errors gracefully."""
        self.logger.error(f"Error in {agent_name}: {str(error)}")
        return {
            "success": False,
            "error": f"{agent_name} error: {str(error)}",
            "agent": agent_name
        }

# Make sure the class is exported
__all__ = ['MasterOrchestratorAgent']
