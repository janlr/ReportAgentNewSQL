import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import openai
from pathlib import Path
from agents.base_agent import BaseAgent
from sqlalchemy import create_engine

class ReportGeneratorAgent(BaseAgent):
    """Agent responsible for generating reports."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str, openai_api_key: str):
        """Initialize with configuration."""
        super().__init__("report_generator_agent")
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.openai_api_key = openai_api_key
        
    async def initialize(self) -> bool:
        """Initialize the report generator."""
        try:
            # Set up OpenAI API key
            openai.api_key = self.openai_api_key
            
            # Verify output directory exists and is writable
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)
            
            # Test write permissions
            test_file = self.output_dir / ".test"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                self.logger.error(f"Output directory is not writable: {str(e)}")
                return False
            
            self.logger.info("Report generator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing report generator: {str(e)}")
            return False
    
    async def cleanup(self) -> bool:
        """Clean up resources."""
        try:
            # Any cleanup needed
            self.logger.info("Report generator cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up report generator: {str(e)}")
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process report generation requests."""
        try:
            if not self.validate_input(input_data, ["action", "parameters"]):
                raise ValueError("Missing required fields: action, parameters")
            
            action = input_data["action"]
            parameters = input_data["parameters"]
            
            if action == "create_visualization":
                # Get data if not provided
                if "data" not in parameters:
                    data_tables = await self._gather_report_data(parameters)
                    if not data_tables or "sales_data" not in data_tables:
                        raise ValueError("Failed to gather visualization data")
                    data = data_tables["sales_data"]
                else:
                    data = parameters["data"]
                
                # Create visualization
                return await self._create_visualization(
                    data,
                    parameters["viz_type"],
                    parameters.get("options", {})
                )
            
            elif action == "generate_report":
                # Gather data
                data_tables = await self._gather_report_data(parameters)
                
                # Generate charts
                charts = await self._generate_charts(data_tables, parameters)
                
                # Generate insights if requested
                insights = None
                if parameters.get("include_insights", True):
                    insights = await self._generate_insights(data_tables, parameters)
                
                # Generate summary
                summary = await self._generate_summary(data_tables, parameters)
                
                # Generate output file
                output_file = await self._generate_output_file(
                    data_tables, charts, insights, summary, parameters
                )
                
                return {
                    "success": True,
                    "data": {
                        "data_tables": data_tables,
                        "charts": charts,
                        "insights": insights,
                        "summary": summary,
                        "output_file": output_file
                    }
                }
            
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing report request: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

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
