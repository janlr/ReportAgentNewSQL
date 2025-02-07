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

class MasterOrchestratorAgent(BaseAgent):
    """Agent responsible for coordinating all other agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        super().__init__("master_orchestrator_agent")
        self.config = config
        
        # Initialize agent instances
        self.agents = {
            "database": DatabaseAgent(config["database"]),
            "data_manager": DataManagerAgent(config["data_manager"]),
            "user_interface": UserInterfaceAgent(config["user_interface"]),
            "report_generator": ReportGeneratorAgent(
                config=config["report_generator"]["config"],
                output_dir=config["report_generator"]["output_dir"],
                openai_api_key=config["report_generator"]["openai_api_key"]
            ),
            "insight_generator": InsightGeneratorAgent(config["insight_generator"])
        }
        self.workflow_history = []
    
    async def initialize(self) -> bool:
        """Initialize all agents."""
        try:
            self.logger.info("Initializing database agent...")
            if not await self.agents["database"].initialize():
                self.logger.error("Failed to initialize database agent")
                return False
            
            self.logger.info("Initializing data manager agent...")
            if not await self.agents["data_manager"].initialize():
                self.logger.error("Failed to initialize data manager agent")
                return False
            
            self.logger.info("Initializing user interface agent...")
            if not await self.agents["user_interface"].initialize():
                self.logger.error("Failed to initialize user interface agent")
                return False
            
            self.logger.info("Initializing report generator agent...")
            if not await self.agents["report_generator"].initialize():
                self.logger.error("Failed to initialize report generator agent")
                return False
            
            self.logger.info("Initializing insight generator agent...")
            if not await self.agents["insight_generator"].initialize():
                self.logger.error("Failed to initialize insight generator agent")
                return False
            
            self.logger.info("All agents initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in master orchestrator initialization: {str(e)}")
            return False
    
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
        """Handle report generation workflow."""
        try:
            # Extract parameters
            report_type = input_data.get("report_type")
            parameters = input_data.get("parameters", {})
            generate_summary = input_data.get("generate_summary", False)
            
            # Get data from database
            db_result = await self.agents["database"].process({
                "action": "execute_query",
                "query": parameters.get("query"),
                "params": parameters.get("query_params", {})
            })
            
            # Clean and transform data
            data_result = await self.agents["data_manager"].process({
                "action": "clean_data",
                "data": db_result["data"]
            })
            
            # Generate report
            report_result = await self.agents["report_generator"].process({
                "workflow": "report_generation",
                "action": "generate_report",
                "parameters": {
                    "report_type": report_type,
                    "data": data_result["data"],
                    "options": parameters
                }
            })
            
            # Generate insights if requested
            if generate_summary:
                insight_result = await self.agents["insight_generator"].process({
                    "action": "generate_summary",
                    "data": data_result["data"],
                    "metadata": {
                        "report_type": report_type,
                        "parameters": parameters
                    }
                })
                
                report_result["data"]["insights"] = insight_result["data"]
            
            # Log workflow completion
            self.workflow_history.append({
                "workflow": "report_generation",
                "report_type": report_type,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
            
            return report_result
            
        except Exception as e:
            self.logger.error(f"Error in report generation workflow: {str(e)}")
            
            # Log workflow failure
            self.workflow_history.append({
                "workflow": "report_generation",
                "report_type": report_type,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            raise
    
    async def _handle_data_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data analysis workflow."""
        try:
            action = input_data.get("action")
            parameters = input_data.get("parameters", {})
            
            if action == "get_schema_info":
                return await self.agents["database"].process({
                    "action": "get_schema_info"
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
                return await self.agents["database"].process({
                    "action": "execute_query",
                    "query": parameters.get("query"),
                    "params": parameters.get("params", {})
                })
            
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

    async def _handle_agent_error(self, agent_name: str, error: Exception) -> Dict[str, Any]:
        """Handle agent errors gracefully."""
        self.logger.error(f"Error in {agent_name}: {str(error)}")
        return {
            "success": False,
            "error": f"{agent_name} error: {str(error)}",
            "agent": agent_name
        } 