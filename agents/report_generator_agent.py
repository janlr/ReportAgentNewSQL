import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import openai
from pathlib import Path
from sqlalchemy import create_engine
from .base_agent import BaseAgent
from autogen import UserProxyAgent, AssistantAgent
from .data_manager_agent import DataManagerAgent
from .insight_generator_agent import InsightGeneratorAgent
from .llm_manager_agent import LLMManagerAgent

# Import specific agents
from agents import MasterOrchestratorAgent

# Import all agents
from agents import *

# Import the package
import agents

class ReportGeneratorAgent(BaseAgent):
    """Agent responsible for generating reports."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        super().__init__(config)
        self.required_config = ["template_dir", "output_dir"]
        self.template_dir = Path(config.get("template_dir", "./templates"))
        self.output_dir = Path(config.get("output_dir", "./reports"))
        
        # Initialize the agent system
        self.analyst_agent = AssistantAgent(
            name="analyst",
            system_message="You are an expert data analyst. You analyze data and generate insights.",
            llm_config={"temperature": 0.7}
        )
        
        self.visualization_agent = AssistantAgent(
            name="visualizer",
            system_message="You are an expert in data visualization. You create effective and informative visualizations.",
            llm_config={"temperature": 0.4}
        )
        
        self.insight_agent = AssistantAgent(
            name="insight_generator",
            system_message="You are an expert in interpreting data analysis and generating business insights.",
            llm_config={"temperature": 0.7}
        )
        
        self.coordinator = UserProxyAgent(
            name="coordinator",
            system_message="You coordinate the report generation process between different agents.",
            human_input_mode="NEVER"
        )
        
    async def initialize(self) -> bool:
        """Initialize the report generator agent."""
        if not self.validate_config(self.required_config):
            return False
            
        # Create necessary directories
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process report generation requests."""
        try:
            action = request.get("action")
            if not action:
                return {"success": False, "error": "No action specified"}
            
            if action == "generate_report":
                return await self._generate_report(request.get("parameters", {}))
            elif action == "generate_report_from_prompt":
                return await self._generate_report_from_prompt(request.get("parameters", {}))
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            self.logger.error(f"Error processing report request: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report based on parameters."""
        # TODO: Implement actual report generation
        return {
            "success": True,
            "data": {
                "summary": "Report summary",
                "charts": [],
                "insights": [],
                "data_tables": {},
                "report_url": ""
            }
        }
    
    async def _generate_report_from_prompt(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report from a natural language prompt."""
        # TODO: Implement prompt-based report generation
        return {
            "success": True,
            "data": {
                "summary": "Report summary",
                "charts": [],
                "insights": [],
                "data_tables": {},
                "report_url": ""
            }
        }

    async def cleanup(self) -> bool:
        """Clean up resources."""
        try:
            # Any cleanup needed
            self.logger.info("Report generator cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up report generator: {str(e)}")
            return False
    
    async def _gather_report_data(self, parameters: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Gather data for the report based on parameters."""
        try:
            # Get schema mapping from config
            from . import get_config
            db_config = get_config('database')
            schema_mapping = db_config.get('schema_mapping', {}).get('sales', {})
            column_mapping = schema_mapping.get('column_mapping', {})
            
            # Build query dynamically based on schema mapping
            def build_query(viz_type: str) -> str:
                if viz_type == "Sales Trends":
                    return f"""
                        SELECT 
                            CAST({schema_mapping['orders_table']}.{column_mapping['order_date']} AS DATE) as OrderDate,
                            SUM({schema_mapping['order_details_table']}.{column_mapping['total_amount']}) as TotalSales,
                            COUNT(DISTINCT {schema_mapping['orders_table']}.{column_mapping['order_id']}) as OrderCount
                        FROM {schema_mapping['orders_table']}
                        JOIN {schema_mapping['order_details_table']} 
                            ON {schema_mapping['orders_table']}.{column_mapping['order_id']} = 
                               {schema_mapping['order_details_table']}.{column_mapping['order_id']}
                        WHERE {schema_mapping['orders_table']}.{column_mapping['order_date']} 
                            BETWEEN @start_date AND @end_date
                        GROUP BY CAST({schema_mapping['orders_table']}.{column_mapping['order_date']} AS DATE)
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
                        ORDER BY SUM(SOD.LineTotal) DESC
                    """
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
            
            # Build the query
            viz_type = parameters.get("viz_type")
            query = build_query(viz_type)

            # Execute query
            engine = create_engine(self._build_connection_string(db_config))
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, params=parameters)
            
            return {"sales_data": df}

        except Exception as e:
            self.logger.error(f"Error gathering report data: {str(e)}")
            raise

    def _build_connection_string(self, config: Dict[str, Any]) -> str:
        """Build database connection string."""
        return (
            f"mssql+pyodbc://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['database']}?"
            f"driver={config['driver']}"
        )

    async def _generate_charts(self, data_tables: Dict[str, pd.DataFrame], parameters: Dict[str, Any]) -> List[go.Figure]:
        """Generate charts based on the data and parameters."""
        try:
            charts = []
            report_type = parameters.get("report_type")
            
            if report_type == "Sales Analysis":
                sales_data = data_tables.get("sales_data")
                if sales_data is not None:
                    # Sales over time
                    fig_time = px.line(
                        sales_data,
                        x="OrderDate",
                        y="LineTotal",
                        title="Sales Over Time"
                    )
                    charts.append(fig_time)
                    
                    # Sales by category
                    fig_category = px.bar(
                        sales_data.groupby("CategoryName")["LineTotal"].sum().reset_index(),
                        x="CategoryName",
                        y="LineTotal",
                        title="Sales by Category"
                    )
                    charts.append(fig_category)
                    
                    # Sales by territory
                    fig_territory = px.pie(
                        sales_data.groupby("Territory")["LineTotal"].sum().reset_index(),
                        values="LineTotal",
                        names="Territory",
                        title="Sales by Territory"
                    )
                    charts.append(fig_territory)
            
            elif report_type == "Customer Insights":
                # Similar implementation for customer insights charts
                pass
                
            elif report_type == "Inventory Status":
                # Similar implementation for inventory status charts
                pass
                
            elif report_type == "Financial Performance":
                # Similar implementation for financial performance charts
                pass
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {str(e)}")
            raise

    async def _generate_insights(self, data_tables: Dict[str, pd.DataFrame], parameters: Dict[str, Any]) -> Optional[str]:
        """Generate insights from the data."""
        try:
            if not self.openai_api_key:
                return None
            
            report_type = parameters.get("report_type")
            insights = []
            
            if report_type == "Sales Analysis":
                sales_data = data_tables.get("sales_data")
                if sales_data is not None:
                    # Calculate key metrics
                    total_sales = sales_data["LineTotal"].sum()
                    avg_order_value = sales_data["LineTotal"].mean()
                    top_categories = sales_data.groupby("CategoryName")["LineTotal"].sum().nlargest(3)
                    top_territories = sales_data.groupby("Territory")["LineTotal"].sum().nlargest(3)
                    
                    # Generate insights using OpenAI
                    prompt = f"""
                    Analyze the following sales metrics and provide 3-5 key business insights:
                    
                    Total Sales: ${total_sales:,.2f}
                    Average Order Value: ${avg_order_value:,.2f}
                    
                    Top Categories:
                    {top_categories.to_string()}
                    
                    Top Territories:
                    {top_territories.to_string()}
                    
                    Provide insights in bullet points.
                    """
                    
                    response = await openai.ChatCompletion.acreate(
                        model="gpt-4-turbo-preview",
                        messages=[
                            {"role": "system", "content": "You are a business analyst providing insights from sales data."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    return response.choices[0].message.content
            
            elif report_type == "Customer Insights":
                # Similar implementation for customer insights
                pass
                
            elif report_type == "Inventory Status":
                # Similar implementation for inventory status
                pass
                
            elif report_type == "Financial Performance":
                # Similar implementation for financial performance
                pass
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            return None

    async def _generate_summary(self, data_tables: Dict[str, pd.DataFrame], parameters: Dict[str, Any]) -> Optional[str]:
        """Generate a summary of the report."""
        try:
            if not self.openai_api_key:
                return None
            
            report_type = parameters.get("report_type")
            summary_data = {}
            
            if report_type == "Sales Analysis":
                sales_data = data_tables.get("sales_data")
                if sales_data is not None:
                    summary_data.update({
                        "total_sales": sales_data["LineTotal"].sum(),
                        "total_orders": len(sales_data["OrderDate"].unique()),
                        "avg_order_value": sales_data["LineTotal"].mean(),
                        "top_category": sales_data.groupby("CategoryName")["LineTotal"].sum().idxmax(),
                        "top_territory": sales_data.groupby("Territory")["LineTotal"].sum().idxmax()
                    })
            
            # Generate summary using OpenAI
            prompt = f"""
            Generate a brief executive summary for a {report_type} report with the following metrics:
            
            {json.dumps(summary_data, indent=2)}
            
            Keep the summary concise and focused on key findings.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a business analyst writing executive summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return None

    async def _generate_output_file(
        self,
        data_tables: Dict[str, pd.DataFrame],
        charts: List[go.Figure],
        insights: Optional[str],
        summary: Optional[str],
        parameters: Dict[str, Any]
    ) -> str:
        """Generate output file in the specified format."""
        try:
            output_format = parameters.get("output_format", "Interactive Dashboard")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_type = parameters.get("report_type", "Report").replace(" ", "_")
            filename = f"{report_type}_{timestamp}"
            
            if output_format == "PDF Report":
                # TODO: Implement PDF generation
                pass
                
            elif output_format == "Excel Spreadsheet":
                excel_path = self.output_dir / f"{filename}.xlsx"
                with pd.ExcelWriter(excel_path) as writer:
                    # Write data tables
                    for name, df in data_tables.items():
                        df.to_excel(writer, sheet_name=name, index=False)
                    
                    # Write summary and insights
                    summary_df = pd.DataFrame({
                        "Section": ["Summary", "Insights"],
                        "Content": [summary or "", insights or ""]
                    })
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                return str(excel_path)
                
            elif output_format == "HTML Report":
                # TODO: Implement HTML report generation
                pass
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error generating output file: {str(e)}")
            return ""

    async def _create_visualization(self, data: pd.DataFrame, viz_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization based on type and options."""
        try:
            if viz_type == "Sales Trends":
                fig = px.line(
                    data,
                    x="OrderDate",
                    y="TotalSales",
                    title="Sales Trends Over Time"
                )
            elif viz_type == "Product Performance":
                fig = px.bar(
                    data,
                    x="ProductName",
                    y=options.get("metric", "Revenue"),
                    title=f"Top Products by {options.get('metric', 'Revenue')}"
                )
            elif viz_type == "Customer Segments":
                fig = px.scatter(
                    data,
                    x="OrderCount",
                    y="AvgOrderValue",
                    size="TotalSpent",
                    title="Customer Segmentation"
                )
            elif viz_type == "Geographic Analysis":
                fig = px.choropleth(
                    data,
                    locations="Region",
                    color=options.get("metric", "TotalSales"),
                    title=f"Geographic Distribution of {options.get('metric', 'Sales')}"
                )
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            return {
                "success": True,
                "data": {
                    "chart": fig
                }
            }
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def generate_report(self, query: str) -> Dict[str, Any]:
        """
        Generates any type of report based on natural language query and available data
        using a multi-agent approach
        
        Args:
            query (str): Natural language query describing the desired report
            
        Returns:
            Dict containing:
            - analysis_results: Dict of analyzed data
            - visualizations: List of generated visualizations
            - insights: Natural language insights
            - metadata: Report generation metadata
        """
        try:
            # Get available data schema
            available_data = await self._get_available_data_schema()
            
            # Initialize the report generation chat
            report_request = {
                'query': query,
                'available_data': available_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Start the multi-agent conversation
            await self.coordinator.initiate_chat(
                self.analyst_agent,
                message=f"Analyze this report request: {report_request}"
            )
            
            # Validate feasibility
            feasibility_result = await self._validate_request(query, available_data)
            if not feasibility_result['is_feasible']:
                return {
                    'error': 'Cannot generate requested report with available data',
                    'suggestions': feasibility_result['alternative_suggestions'],
                    'status': 'failed',
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'query': query
                    }
                }
            
            # Generate the report components using the agent system
            analysis_results = await self._generate_analysis(
                feasibility_result['validated_params'],
                available_data
            )
            
            visualizations = await self._generate_visualizations(
                analysis_results,
                feasibility_result['validated_params']
            )
            
            insights = await self._generate_insights(
                analysis_results,
                visualizations,
                feasibility_result['validated_params']
            )
            
            return {
                'analysis_results': analysis_results,
                'visualizations': visualizations,
                'insights': insights,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'query': query,
                    'data_columns_used': list(analysis_results.keys()),
                    'status': 'success'
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed',
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'query': query
                }
            }

    async def _validate_request(self, query: str, available_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses the analyst agent to validate if the report request is feasible
        """
        validation_result = await self.coordinator.get_agent_response(
            self.analyst_agent,
            f"Validate if this report request can be fulfilled with available data: {query}\nAvailable data: {available_data}"
        )
        return validation_result

    async def _generate_analysis(self, params: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses the analyst agent to generate the analysis
        """
        analysis_result = await self.coordinator.get_agent_response(
            self.analyst_agent,
            f"Generate analysis based on these parameters: {params}\nUsing data: {data}"
        )
        return analysis_result

    async def _generate_visualizations(self, analysis_results: Dict[str, Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Uses the visualization agent to generate appropriate visualizations
        """
        viz_result = await self.coordinator.get_agent_response(
            self.visualization_agent,
            f"Create visualizations for these analysis results: {analysis_results}\nParameters: {params}"
        )
        return viz_result

    async def _generate_insights(self, analysis_results: Dict[str, Any], visualizations: List[Dict[str, Any]], params: Dict[str, Any]) -> List[str]:
        """
        Uses the insight agent to generate insights from the analysis and visualizations
        """
        insights_result = await self.coordinator.get_agent_response(
            self.insight_agent,
            f"Generate insights from analysis: {analysis_results}\nVisualizations: {visualizations}\nParameters: {params}"
        )
        return insights_result

    async def _get_available_data_schema(self) -> Dict[str, Any]:
        """
        Gets the available data schema from the data source
        """
        return await self.data_source.get_schema()
