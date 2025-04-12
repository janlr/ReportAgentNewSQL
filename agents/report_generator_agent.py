import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import openai
from pathlib import Path
from sqlalchemy import create_engine
import asyncio
from .base_agent import BaseAgent
from autogen import UserProxyAgent, AssistantAgent
import numpy as np
from anthropic import Anthropic

class ReportGeneratorAgent(BaseAgent):
    """Agent responsible for generating reports."""
    
    def __init__(self, report_config: Dict[str, Any], db_config: Dict[str, Any], output_dir: str, anthropic_api_key: str, orchestrator: Any):
        """Initialize the report generator."""
        # Pass the specific report_config to the BaseAgent constructor
        super().__init__("report_generator_agent", report_config) 
        self.db_config = db_config # Store database configuration
        self.output_dir = Path(output_dir)
        self.anthropic_api_key = anthropic_api_key
        self.orchestrator = orchestrator
        # Use the correct key 'templates_dir' from the report_config
        self.template_dir = Path(report_config.get("templates_dir", "./templates"))
        # Only require 'templates_dir' in the report_config dict
        self.required_config = ["templates_dir"]
        self.client: Optional[Anthropic] = None # Initialize client to None
        
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
        """Initialize the report generator."""
        if not self.validate_config(self.required_config):
            return False
            
        try:
            # Set up Anthropic API key using the correct class
            self.logger.info("Initializing Anthropic client...")
            if not self.anthropic_api_key:
                self.logger.error("Anthropic API key is missing!")
                raise ValueError("Anthropic API key not provided")
            # Use the imported Anthropic class
            self.client = Anthropic(api_key=self.anthropic_api_key)
            # Debug print to confirm type
            self.logger.info(f"Anthropic client initialized. Type: {type(self.client)}") 
            
            # Create necessary directories
            self.template_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = self.output_dir / ".test_write"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                self.logger.error(f"Output directory is not writable: {str(e)}", exc_info=True)
                return False
            
            self.logger.info("Report generator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing report generator: {str(e)}", exc_info=True)
            return False
    
    async def cleanup(self) -> bool:
        """Clean up resources."""
        try:
            # Clean up any resources
            self.logger.info("Report generator cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up report generator: {str(e)}")
            return False
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process report generation requests."""
        try:
            action = request.get("action")
            parameters = request.get("parameters", {})
            if not action or not parameters:
                # Allow action without parameters for some cases? Check specific actions.
                if action not in ["some_action_without_params"]: # Example placeholder
                     raise ValueError("Missing required fields: action or parameters for this action")
                elif not action:
                     raise ValueError("Missing required field: action")
            
            self.logger.info(f"Processing report request - Action: {action}")

            if action == "create_visualization":
                # Simplified for now, assumes data is passed or handled elsewhere
                # Ideally, data gathering would happen here if needed.
                data = parameters.get("data")
                if data is None:
                     # Maybe try gathering data based on viz_type if data not provided?
                     # For now, require data for simplicity
                     return {"success": False, "error": "Data required for create_visualization"}
                return await self._create_visualization(
                    pd.DataFrame(data), # Ensure it's a DataFrame
                    parameters.get("viz_type", "Unknown"),
                    parameters.get("options", {})
                )
            
            elif action == "generate_report":
                # Call the new handler for structured reports from advanced options
                return await self._generate_report_structured(parameters)

            elif action == "generate_report_from_prompt":
                return await self._generate_report_from_prompt(parameters)
            
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing report request: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Error in ReportGeneratorAgent.process: {str(e)}"}
    
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
            if not self.anthropic_api_key:
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
                    
                    # Generate insights using Anthropic
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
                    
                    response = await self.client.generate_content(prompt)
                    
                    return response
            
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
            if not self.anthropic_api_key:
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
            
            # Generate summary using Anthropic
            prompt = f"""
            Generate a brief executive summary for a {report_type} report with the following metrics:
            
            {json.dumps(summary_data, indent=2)}
            
            Keep the summary concise and focused on key findings.
            """
            
            response = await self.client.generate_content(prompt)
            
            return response
            
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

    async def _generate_report_from_prompt(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report from a natural language prompt."""
        prompt = parameters.get("prompt")
        include_charts = parameters.get("include_charts", True)
        include_insights = parameters.get("include_insights", True)
        output_format = parameters.get("output_format", "Interactive Dashboard")

        if not prompt:
            return {"success": False, "error": "No prompt provided"}
        if not self.orchestrator:
             return {"success": False, "error": "Orchestrator not available for report generation"}
        if not self.client:
             return {"success": False, "error": "Anthropic client not initialized"}
        if not self.db_config:
             return {"success": False, "error": "Database configuration not available"}

        self.logger.info(f"Generating report from prompt: '{prompt}'")
        start_time = datetime.now()

        try:
            # --- Step 1: Get Schema Summary --- 
            self.logger.info("Step 1: Getting DB Schema Summary...")
            schema_summary = await self._get_db_schema_summary()
            if schema_summary.startswith("Error:"):
                self.logger.error(f"Failed Step 1: {schema_summary}")
                return {"success": False, "error": schema_summary}
            self.logger.info("Step 1: Completed.")

            # --- Step 2: Construct SQL Generation Prompt ---
            self.logger.info("Step 2: Constructing SQL Generation Prompt...")
            
            db_type = self.db_config.get('type', 'unknown').lower()
            if db_type == 'mssql':
                db_name = "MS SQL Server"
                limit_clause_instruction = "use the `TOP N` syntax (e.g., `SELECT TOP 10 ...`), NOT the `LIMIT` clause."
            elif db_type == 'mysql':
                db_name = "MySQL"
                limit_clause_instruction = "use the `LIMIT N` clause (e.g., `... ORDER BY col LIMIT 10`), NOT `TOP N`."
            elif db_type == 'postgresql':
                db_name = "PostgreSQL"
                limit_clause_instruction = "use the `LIMIT N` clause (e.g., `... ORDER BY col LIMIT 10`), NOT `TOP N`."
            else:
                db_name = "SQL"
                limit_clause_instruction = "use the standard `LIMIT N` clause if limiting results."
                self.logger.warning(f"Unknown DB type '{db_type}', defaulting to standard LIMIT clause instruction.")

            sql_prompt_template = """Relevant Database Schema:
{schema_summary}
User Request: {user_prompt}

Generate a single, executable {db_name} query to answer the user's request based *only* on the schema provided.

Rules:
- Only output the raw SQL query, with no explanations or markdown formatting surrounding it.
- Ensure the query is valid for {db_name}.
- Use standard SQL syntax for {db_name}.
- If the request requires limiting the number of results (e.g., top 10, first 5), {limit_instruction}
- Infer join conditions from common column names (like CustomerID, ProductID, SalesOrderID, TerritoryID) if necessary.

SQL Query:"""
            sql_generation_prompt = sql_prompt_template.format(
                schema_summary=schema_summary,
                user_prompt=prompt, 
                db_name=db_name, 
                limit_instruction=limit_clause_instruction
            )
            self.logger.info(f"Step 2: Completed (Prompt tailored for {db_name}).")

            # --- Step 3: Generate SQL using LLM ---
            self.logger.info("Step 3: Generating SQL query using LLM (Haiku)...")
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.1, 
                messages=[{"role": "user", "content": sql_generation_prompt}]
            )
            generated_sql = ""
            if response.content:
                for block in response.content:
                    if block.type == 'text':
                        generated_sql += block.text
            generated_sql = generated_sql.strip().replace('```sql', '').replace('```', '').strip()

            if not generated_sql:
                self.logger.error("LLM failed to generate SQL query.")
                return {"success": False, "error": "Failed to generate SQL query from prompt"}
            self.logger.info(f"Generated SQL:\n{generated_sql}")
            self.logger.info("Step 3: Completed.")

            # --- Step 4: Execute Generated SQL --- 
            self.logger.info("Step 4: Executing generated SQL query...")
            db_result = await self.orchestrator.process({
                "workflow": "data_analysis",
                "action": "execute_query",
                "parameters": {"query": generated_sql}
            })
            
            if not db_result.get("success"):
                error_msg = db_result.get('error', 'Unknown database error')
                self.logger.error(f"Failed to execute generated SQL: {error_msg}\nSQL: {generated_sql}")
                return {"success": False, "error": f"Failed to execute generated SQL: {error_msg}\nSQL Used: {generated_sql}"}
            self.logger.info("Step 4: Completed.")
                
            # Process data into DataFrame
            data = db_result.get("data", [])
            if not data:
                self.logger.warning("Generated query executed successfully but returned no data.")
                return {
                    "success": True,
                    "data": {
                        "summary": f"Query executed successfully but returned no data.\nPrompt: {prompt}\nSQL: {generated_sql}",
                        "charts": [],
                        "insights": "No data available for insights.",
                        "data_tables": {"query_results": pd.DataFrame()},
                        "generated_sql": generated_sql,
                        "report_url": ""
                    }
                }
            
            df = pd.DataFrame(data)
            try: 
                df = df.infer_objects()
            except Exception as type_err:
                self.logger.warning(f"Could not infer object types in DataFrame: {type_err}")
            df = df.fillna('N/A') 
            self.logger.info(f"Query executed successfully, retrieved {len(df)} rows.")
            data_tables = {"query_results": df}

            # --- Step 5: Generate Summary, Insights, Charts ---
            self.logger.info("Step 5: Generating Summary, Insights, and Charts...")
            summary = "Summary generation skipped by default."
            insights = "Insights generation skipped by default."
            charts = []

            # Generate Summary & Insights via LLM
            if include_insights:
                self.logger.info("Generating summary and insights using LLM (Sonnet)...")
                # Construct prompt based *only* on info available in this flow
                data_sample_str = df.head(10).to_string() # Show more rows for context

                analysis_prompt = (
                    f"User Request: {prompt}\n\n" +
                    f"Generated SQL Query:\n{generated_sql}\n\n" +
                    f"Result Data Sample (first 10 rows):\n{data_sample_str}\n\n" +
                    f"Based ONLY on the user request and the resulting data sample, provide:\n" +
                    f"1. SUMMARY: A concise summary of the key findings directly answering the user request.\n" +
                    f"2. INSIGHTS: Bullet points of 1-3 interesting patterns or observations found ONLY in the provided data sample.\n\n" +
                    f"Respond ONLY with SUMMARY: and INSIGHTS: sections."
                )

                try:
                    analysis_response = self.client.messages.create(
                        model="claude-3-sonnet-20240229", # Or your preferred model
                        max_tokens=800,
                        temperature=0.4,
                        messages=[{"role": "user", "content": analysis_prompt}]
                    )
                    analysis_text = ""
                    if analysis_response.content:
                        for block in analysis_response.content:
                            if block.type == 'text':
                                analysis_text += block.text
                    
                    # Extract Summary and Insights
                    summary_part = analysis_text.split("INSIGHTS:")[0].replace("SUMMARY:", "").strip()
                    insights_part = analysis_text.split("INSIGHTS:")[1].strip() if "INSIGHTS:" in analysis_text else "No specific insights generated."
                    summary = summary_part if summary_part else "Summary could not be generated."
                    insights = insights_part if insights_part else "No specific insights generated."
                    self.logger.info("Summary and insights generated.")
                    
                except Exception as llm_err:
                    self.logger.error(f"LLM call for summary/insights failed: {llm_err}", exc_info=True)
                    summary = f"Error generating summary: {llm_err}"
                    insights = "Error generating insights."
            
            # Basic Chart Generation
            if include_charts and not df.empty:
                self.logger.info("Attempting basic chart generation...")
                try:
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64', 'datetimetz']).columns.tolist()

                    # Prioritize Time Series if applicable
                    if datetime_cols and numeric_cols:
                        try:
                           fig_ts = px.line(df.sort_values(by=datetime_cols[0]), x=datetime_cols[0], y=numeric_cols[0], title=f"{numeric_cols[0]} over Time ({datetime_cols[0]})")
                           charts.append(fig_ts)
                           self.logger.info(f"Generated time series chart.")
                        except Exception as ts_err:
                             self.logger.warning(f"Could not generate time series plot: {ts_err}")
                    
                    # Then Bar Chart
                    if categorical_cols and numeric_cols:
                        cat_col = categorical_cols[0]
                        num_col = numeric_cols[0] 
                        # Avoid plotting high-cardinality categorical data
                        if df[cat_col].nunique() < 50: 
                            try:
                                # Aggregate data for bar chart if necessary (e.g., multiple rows per category)
                                agg_df = df.groupby(cat_col)[num_col].sum().reset_index()
                                fig_bar = px.bar(agg_df, x=cat_col, y=num_col, title=f"Total {num_col} by {cat_col}")
                                charts.append(fig_bar)
                                self.logger.info(f"Generated bar chart.")
                            except Exception as bar_err:
                                self.logger.warning(f"Could not generate bar chart: {bar_err}")
                        else:
                            self.logger.info(f"Skipping bar chart for {cat_col} due to high cardinality.")
                            
                    # Then Scatter Plot
                    if len(numeric_cols) >= 2:
                        try:
                            fig_scatter = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                            charts.append(fig_scatter)
                            self.logger.info(f"Generated scatter plot.")
                        except Exception as scatter_err:
                            self.logger.warning(f"Could not generate scatter plot: {scatter_err}")

                except Exception as chart_err:
                    self.logger.error(f"Failed during chart generation logic: {chart_err}", exc_info=True)
            self.logger.info(f"Step 5: Completed. Generated {len(charts)} charts.")

            # --- Step 6: Generate Output File (Placeholder) ---
            self.logger.info("Step 6: Generating output file (Not Implemented)...")
            # TODO: Implement output file generation based on output_format
            report_url = "" 
            self.logger.info("Step 6: Completed.")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.logger.info(f"Report generation from prompt completed in {duration:.2f} seconds.")

            return {
                "success": True,
                "data": {
                    "summary": summary,
                    "charts": charts,
                    "insights": insights,
                    "data_tables": data_tables, 
                    "generated_sql": generated_sql, 
                    "report_url": report_url
                }
            }

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.logger.error(f"Error generating report from prompt after {duration:.2f}s: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Error generating report from prompt: {type(e).__name__}: {str(e)}"}

    async def _get_db_schema_summary(self) -> str:
        """Retrieves a concise summary of the database schema using the orchestrator."""
        try:
            self.logger.info("Retrieving database schema summary via orchestrator...")
            if not self.orchestrator:
                 return "Error: Orchestrator not available to retrieve schema."
                 
            # Get list of tables first
            schema_info_result = await self.orchestrator.process({
                "workflow": "data_analysis", 
                "action": "get_schema_info"
            })
            
            if not schema_info_result.get("success"):
                error_msg = schema_info_result.get('error', 'Unknown error')
                self.logger.error(f"Failed to get table list via orchestrator: {error_msg}")
                return f"Error: Could not retrieve table list: {error_msg}"
            
            tables = schema_info_result.get("data", {}).get("tables", [])
            if not tables:
                self.logger.warning("Orchestrator returned schema info successfully, but no tables were found.")
                return "Schema information retrieved, but no tables found."

            # Request detailed info for each table to get columns concurrently
            schema_summary = "Available Tables and Columns:\n"
            tasks = []
            valid_tables_for_tasks = [] # Keep track of tables we are fetching details for
            for table_info in tables:
                table_name = table_info.get("name")
                schema_name = table_info.get("schema")
                if table_name and schema_name:
                    valid_tables_for_tasks.append(table_info) # Store table info associated with task
                    tasks.append(self.orchestrator.process({
                        "workflow": "data_analysis",
                        "action": "get_schema_info",
                        "parameters": {"table_name": table_name, "schema_name": schema_name}
                    }))
            
            # Run all detailed info requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results to build the schema summary
            for i, result in enumerate(results):
                if i >= len(valid_tables_for_tasks):
                    self.logger.error("Result index out of bounds for valid tasks.")
                    continue # Should not happen if logic is correct
                
                current_table = valid_tables_for_tasks[i]
                schema_name = current_table.get("schema")
                table_name = current_table.get("name")
                table_key = f"{schema_name}.{table_name}"

                col_names_str = "(Error retrieving columns)" # Default
                if isinstance(result, Exception):
                    self.logger.warning(f"Failed to get column details for {table_key}: {result}")
                elif not result.get("success"):
                    self.logger.warning(f"Failed to get column details for {table_key}: {result.get('error')}")
                else:
                    columns = result.get("data", {}).get("columns", [])
                    if columns:
                        col_names = [col.get("name", "?") for col in columns]
                        col_names_str = ", ".join(col_names)
                    else:
                        col_names_str = "(No columns found)"
                
                schema_summary += f"- {table_key}({col_names_str})\n"
                
            self.logger.info("Successfully retrieved and formatted schema summary.")
            return schema_summary
            
        except Exception as e:
            self.logger.error(f"Error getting schema summary via orchestrator: {str(e)}", exc_info=True)
            return f"Error: An unexpected error occurred while retrieving schema information: {str(e)}"

    # --- New method for structured reports ---
    async def _generate_report_structured(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a report based on structured parameters from the UI."""
        self.logger.info("Generating report from structured parameters...")
        start_time = datetime.now()

        # Extract parameters
        primary_table = parameters.get("primary_table")
        dimensions = parameters.get("dimensions", [])
        metrics = parameters.get("metrics", [])
        filters = parameters.get("filters", {})
        time_column = parameters.get("time_column")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        include_charts = parameters.get("include_charts", True)
        chart_types = parameters.get("chart_types", ["Auto"])
        include_insights = parameters.get("include_insights", True)
        output_format = parameters.get("output_format", "Interactive Dashboard")

        # Basic validation
        if not primary_table or not metrics:
            return {"success": False, "error": "Primary table and at least one metric are required."}
        if not self.orchestrator or not self.client:
             return {"success": False, "error": "Agent dependencies (Orchestrator/LLM Client) not initialized"}

        try:
            # --- Step 1: Build SQL Query --- 
            self.logger.info("Step 1: Building SQL query...")
            # TODO: Implement dynamic SQL query building based on parameters
            # This is a complex step requiring careful construction
            # Placeholder query for now
            select_clause = ", ".join([f"SUM({m}) as Total_{m}" for m in metrics])
            if dimensions:
                select_clause = ", ".join(dimensions) + ", " + select_clause
            
            schema_name, table_name = primary_table.split('.') # Assumes format "schema.table"
            
            # Basic WHERE clause (needs significant improvement for different filter types)
            where_clauses = []
            if time_column and start_date and end_date:
                # Use parameterization for dates if possible with the execution method, otherwise ensure proper quoting
                # Assuming direct string formatting for now, ensure dates are quoted
                where_clauses.append(f"{time_column} BETWEEN '{start_date}' AND '{end_date}'") 
            for col, value in filters.items():
                 # Basic filtering - needs enhancement for ranges, text matching etc.
                 if isinstance(value, str):
                      # Escape single quotes for SQL and format safely
                      sql_value = value.replace("'", "''")
                      where_clauses.append(f"{col} = '{sql_value}'") 
                 elif isinstance(value, (int, float)):
                      where_clauses.append(f"{col} = {value}")
                 # Add handling for other types like boolean if needed

            where_string = " AND ".join(where_clauses) if where_clauses else "1=1" # Use 1=1 if no clauses
            
            group_by_clause = "GROUP BY " + ", ".join(dimensions) if dimensions else ""
            
            built_sql = f"SELECT {select_clause} FROM {schema_name}.{table_name} WHERE {where_string} {group_by_clause};"
            self.logger.info(f"Built SQL: \n{built_sql}")
            self.logger.info("Step 1: Completed.") # Changed from Placeholder SQL

            # --- Step 2: Execute SQL Query --- 
            self.logger.info("Step 2: Executing built SQL query...")
            db_result = await self.orchestrator.process({
                "workflow": "data_analysis",
                "action": "execute_query",
                "parameters": {"query": built_sql}
            })
            if not db_result.get("success"):
                error_msg = db_result.get('error', 'Unknown database error')
                self.logger.error(f"Failed to execute built SQL: {error_msg}\nSQL: {built_sql}")
                return {"success": False, "error": f"Failed to execute SQL: {error_msg}\nSQL Used: {built_sql}"}
            self.logger.info("Step 2: Completed.")

            # Process data
            data = db_result.get("data", [])
            if not data:
                self.logger.warning("Built query executed successfully but returned no data.")
                # Return success but indicate no data found
                return {
                    "success": True,
                    "data": {
                        "summary": f"Query executed successfully but returned no data.\nSQL: {built_sql}",
                        "charts": [], "insights": "No data.", "data_tables": {},
                        "generated_sql": built_sql, "report_url": ""
                    }
                }
            df = pd.DataFrame(data)
            df = df.fillna('N/A')
            data_tables = {"query_results": df}
            self.logger.info(f"Query executed, retrieved {len(df)} rows.")

            # --- Step 3: Generate Summary & Insights (Optional) ---
            self.logger.info("Step 3: Generating Summary and Insights...")
            summary = "Summary generation skipped."
            insights = "Insights generation skipped."
            if include_insights:
                # Pre-calculate strings to avoid complex f-string internals
                dimensions_str = ', '.join(dimensions) if dimensions else 'None'
                metrics_str = ', '.join([f'Total_{m}' for m in metrics])
                # Fix for line 902 linter error
                time_period_str = f"{start_date} to {end_date} (Column: {time_column})" if time_column and start_date and end_date else 'N/A' 
                filters_str = json.dumps(filters) if filters else 'None'
                data_sample_str = df.head(5).to_string()

                # Construct prompt using f-strings with pre-calculated parts
                analysis_prompt = f"""Data Analysis Request:
Primary Table: {primary_table}
Dimensions: {dimensions_str}
Metrics: {metrics_str}
Time Period: {time_period_str}
Filters Applied: {filters_str}

Result Data Sample (first 5 rows):
{data_sample_str}

Based ONLY on the request parameters and the data sample, provide:
1. SUMMARY: A concise summary of the key findings related to the request.
2. INSIGHTS: Bullet points of 1-3 interesting patterns observed.

Respond ONLY with SUMMARY: and INSIGHTS: sections."""
                
                self.logger.info("Generating summary and insights using LLM (Sonnet)...")
                try:
                    # ... (LLM call and parsing) ...
                    analysis_response = self.client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=800,
                        temperature=0.4,
                        messages=[{"role": "user", "content": analysis_prompt}]
                    )
                    analysis_text = ""
                    if analysis_response.content:
                        for block in analysis_response.content:
                            if block.type == 'text':
                                analysis_text += block.text
                    summary_part = analysis_text.split("INSIGHTS:")[0].replace("SUMMARY:", "").strip()
                    insights_part = analysis_text.split("INSIGHTS:")[1].strip() if "INSIGHTS:" in analysis_text else "No specific insights generated."
                    summary = summary_part if summary_part else "Summary could not be generated."
                    insights = insights_part if insights_part else "No specific insights generated."
                    self.logger.info("Summary and insights generated.")
                except Exception as llm_err:
                    self.logger.error(f"LLM call for summary/insights failed: {llm_err}", exc_info=True)
                    summary = f"Error generating summary: {llm_err}"
                    insights = "Error generating insights."
            self.logger.info("Step 3: Completed.")

            # --- Step 4: Generate Charts (Optional) ---
            self.logger.info("Step 4: Generating Charts...")
            charts = []
            if include_charts and not df.empty:
                if "Auto" in chart_types or not chart_types:
                    # TODO: Implement Auto chart logic (reuse from prompt-based?)
                    self.logger.info("Attempting Auto chart generation...")
                    # Placeholder: charts.append(some_auto_chart_function(df))
                else:
                    self.logger.info(f"Generating requested chart types: {chart_types}")
                    for chart_type in chart_types:
                        # TODO: Implement specific chart generation based on type,
                        # using dimensions/metrics. Add error handling.
                        # Placeholder:
                        # if chart_type == "Bar Chart": ...
                        # elif chart_type == "Line Chart": ...
                        pass 
            self.logger.info(f"Step 4: Completed. Generated {len(charts)} charts.")

            # --- Step 5: Generate Output File (Placeholder) ---
            self.logger.info("Step 5: Generating output file (Not Implemented)...")
            # TODO: Implement based on output_format
            report_url = "" 
            self.logger.info("Step 5: Completed.")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.logger.info(f"Structured report generation completed in {duration:.2f} seconds.")

            return {
                "success": True,
                "data": {
                    "summary": summary,
                    "charts": charts,
                    "insights": insights,
                    "data_tables": data_tables,
                    "generated_sql": built_sql, # Include the SQL we built
                    "report_url": report_url
                }
            }

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.logger.error(f"Error during structured report generation after {duration:.2f}s: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Error generating structured report: {type(e).__name__}: {str(e)}"}
