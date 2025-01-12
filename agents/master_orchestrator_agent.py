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
        super().__init__("master_orchestrator_agent")
        self.config = config
        
        # Initialize agents
        self.agents = {}
        self.workflow_history = []
    
    async def initialize(self) -> bool:
        """Initialize all agents."""
        try:
            self.logger.info("Initializing master orchestrator")
            
            # Initialize agents
            self.agents["database"] = DatabaseAgent(self.config.get("database", {}))
            self.agents["data_manager"] = DataManagerAgent(self.config.get("data_manager", {}))
            self.agents["user_interface"] = UserInterfaceAgent(self.config.get("user_interface", {}))
            self.agents["report_generator"] = ReportGeneratorAgent(self.config.get("report_generator", {}))
            self.agents["insight_generator"] = InsightGeneratorAgent(self.config.get("insight_generator", {}))
            
            # Initialize each agent
            for name, agent in self.agents.items():
                success = await agent.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize {name} agent")
                    return False
                self.logger.info(f"Initialized {name} agent")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing master orchestrator: {str(e)}")
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process workflow requests."""
        workflow = input_data.get("workflow")
        
        if not workflow:
            raise ValueError("Workflow not specified in input data")
        
        try:
            if workflow == "report_generation":
                return await self._handle_report_generation(input_data)
            
            elif workflow == "data_analysis":
                return await self._handle_data_analysis(input_data)
            
            elif workflow == "database_management":
                return await self._handle_database_management(input_data)
            
            else:
                raise ValueError(f"Unknown workflow: {workflow}")
                
        except Exception as e:
            self.logger.error(f"Error processing workflow request: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up all agents."""
        self.logger.info("Cleaning up master orchestrator")
        
        for name, agent in self.agents.items():
            try:
                await agent.cleanup()
                self.logger.info(f"Cleaned up {name} agent")
            except Exception as e:
                self.logger.error(f"Error cleaning up {name} agent: {str(e)}")
    
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
            
            # Generate visualizations and insights
            components = []
            
            # Add visualizations
            viz_result = await self.agents["report_generator"].process({
                "action": "generate_report",
                "report_type": report_type,
                "data": data_result["data"],
                "parameters": parameters
            })
            components.extend(viz_result.get("components", []))
            
            # Generate summary if requested
            if generate_summary:
                summary_result = await self.agents["insight_generator"].process({
                    "action": "generate_summary",
                    "data": data_result["data"],
                    "metadata": {
                        "report_type": report_type,
                        "parameters": parameters,
                        "query": parameters.get("query")
                    }
                })
                
                # Add summary sections
                components.extend([
                    {
                        "type": "header",
                        "text": "Report Summary"
                    },
                    {
                        "type": "markdown",
                        "text": summary_result["summary"]
                    },
                    {
                        "type": "header",
                        "text": "Key Findings"
                    },
                    {
                        "type": "markdown",
                        "text": summary_result["findings"]
                    },
                    {
                        "type": "header",
                        "text": "Recommendations"
                    },
                    {
                        "type": "markdown",
                        "text": summary_result["recommendations"]
                    }
                ])
            
            # Render the report page
            ui_result = await self.agents["user_interface"].process({
                "action": "render_page",
                "page": f"{report_type.title()} Report",
                "components": components
            })
            
            # Log workflow completion
            self.workflow_history.append({
                "workflow": "report_generation",
                "report_type": report_type,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
            
            return {
                "status": "success",
                "report": ui_result,
                "timestamp": datetime.now().isoformat()
            }
            
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
            # Extract parameters
            data_source = input_data.get("data_source")
            analysis_type = input_data.get("analysis_type")
            parameters = input_data.get("parameters", {})
            
            # Import data if it's a file
            if data_source.get("type") == "file":
                data_result = await self.agents["data_manager"].process({
                    "action": "import_excel",
                    "file_path": data_source["path"],
                    "sheet_name": data_source.get("sheet_name")
                })
            else:
                # Get data from database
                data_result = await self.agents["database"].process({
                    "action": "execute_query",
                    "query": data_source["query"],
                    "params": data_source.get("query_params", {})
                })
            
            # Generate insights
            insight_result = await self.agents["insight_generator"].process({
                "action": "analyze_data",
                "data": data_result["data"],
                "analysis_type": analysis_type,
                "parameters": parameters
            })
            
            # Create visualization components
            components = [
                {
                    "type": "header",
                    "text": f"{analysis_type.title()} Analysis"
                },
                {
                    "type": "markdown",
                    "text": insight_result["summary"]
                }
            ]
            
            # Add visualizations
            components.extend(insight_result.get("visualizations", []))
            
            # Add detailed insights
            components.extend([
                {
                    "type": "header",
                    "text": "Detailed Insights"
                },
                {
                    "type": "markdown",
                    "text": insight_result["details"]
                }
            ])
            
            # Render the analysis page
            ui_result = await self.agents["user_interface"].process({
                "action": "render_page",
                "page": f"{analysis_type.title()} Analysis",
                "components": components
            })
            
            # Log workflow completion
            self.workflow_history.append({
                "workflow": "data_analysis",
                "analysis_type": analysis_type,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
            
            return {
                "status": "success",
                "analysis": ui_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in data analysis workflow: {str(e)}")
            
            # Log workflow failure
            self.workflow_history.append({
                "workflow": "data_analysis",
                "analysis_type": analysis_type,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            raise
    
    async def _handle_database_management(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database management workflow."""
        try:
            # Extract parameters
            action = input_data.get("action")
            parameters = input_data.get("parameters", {})
            
            if action == "get_schema":
                # Get schema information
                schema_result = await self.agents["database"].process({
                    "action": "get_schema_info"
                })
                
                # Create components for schema display
                components = [
                    {
                        "type": "header",
                        "text": "Database Schema"
                    },
                    {
                        "type": "markdown",
                        "text": "### Tables"
                    }
                ]
                
                # Add table information
                for table in schema_result["tables"]:
                    components.extend([
                        {
                            "type": "markdown",
                            "text": f"#### {table['name']}"
                        },
                        {
                            "type": "dataframe",
                            "data": table["columns"]
                        }
                    ])
                
            elif action == "test_connection":
                # Test database connection
                test_result = await self.agents["database"].process({
                    "action": "test_connection"
                })
                
                # Create components for test results
                components = [
                    {
                        "type": "header",
                        "text": "Connection Test Results"
                    }
                ]
                
                if test_result["success"]:
                    components.append({
                        "type": "success",
                        "text": "Database connection successful!"
                    })
                else:
                    components.append({
                        "type": "error",
                        "text": f"Connection failed: {test_result['error']}"
                    })
            
            else:
                raise ValueError(f"Unknown database action: {action}")
            
            # Render the management page
            ui_result = await self.agents["user_interface"].process({
                "action": "render_page",
                "page": "Database Management",
                "components": components
            })
            
            # Log workflow completion
            self.workflow_history.append({
                "workflow": "database_management",
                "action": action,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
            
            return {
                "status": "success",
                "result": ui_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in database management workflow: {str(e)}")
            
            # Log workflow failure
            self.workflow_history.append({
                "workflow": "database_management",
                "action": action,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            raise 