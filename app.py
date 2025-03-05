import streamlit as st
import json
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import pandas as pd
import os

# Import all required agent classes
from agents.master_orchestrator_agent import MasterOrchestratorAgent
from agents.assistant_agent import AssistantAgent
from agents.data_manager_agent import DataManagerAgent
from agents.report_generator_agent import ReportGeneratorAgent
from agents.insight_generator_agent import InsightGeneratorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Report Agent Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Report Agent Dashboard")

# Initialize components
async def init_components():
    try:
        # Load configuration
        config = {
            "database": {
                "type": "mssql",
                "host": "localhost",
                "database": "AdventureWorks2017",
                "driver": "ODBC Driver 17 for SQL Server",
                "echo": True,
                "cache_dir": "./cache"
            },
            "data_manager": {
                "cache_dir": "./cache/data",
                "batch_size": 1000,
                "max_workers": 4,
                "enable_compression": True,
                "retention_days": 30
            },
            "user_interface": {
                "preferences_dir": "./preferences"
            },
            "report_generator": {
                "template_dir": "./templates",
                "output_dir": "./reports",
                "openai_api_key": st.secrets["api_keys"]["openai"]
            },
            "insight_generator": {
                "llm_manager": {
                    "provider": "openai",
                    "model": "gpt-4-turbo-preview",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "api_key": st.secrets["api_keys"]["openai"]
                },
                "cache_dir": "./cache/insights"
            }
        }
        
        # Initialize master orchestrator with required arguments
        orchestrator = MasterOrchestratorAgent(
            config=config,
            output_dir=st.secrets["report_config"]["output_dir"],
            openai_api_key=st.secrets["api_keys"]["openai"]
        )
        success = await orchestrator.initialize()
        
        if not success:
            raise RuntimeError("Failed to initialize master orchestrator")
        
        return orchestrator
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def main():
    st.title("AdventureWorks Business Intelligence Dashboard")
    
    # Initialize session state
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = asyncio.run(init_components())
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["Schema Explorer", "Sales Analysis", "Report Generator", "Data Visualization"]
    )
    
    try:
        if page == "Schema Explorer":
            schema_explorer()
        elif page == "Sales Analysis":
            sales_analysis()
        elif page == "Report Generator":
            report_generator()
        elif page == "Data Visualization":
            data_visualization()
            
    except Exception as e:
        logger.error(f"Error in application: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def schema_explorer():
    st.header("Database Schema Explorer")
    
    # Get schema information through orchestrator
    result = asyncio.run(
        st.session_state.orchestrator.process({
            "workflow": "data_analysis",
            "action": "get_schema_info"
        })
    )
    
    if not result.get("success"):
        st.error(f"Failed to get schema information: {result.get('error')}")
        return
        
    schema_info = result["data"]
    
    # Schema selection
    schemas = sorted(list(set(
        info["schema"] for info in schema_info["tables"].values()
    )))
    selected_schema = st.selectbox("Select Schema", schemas)
    
    # Display tables in selected schema
    st.subheader(f"Tables in {selected_schema} Schema")
    tables = [
        info["name"] 
        for info in schema_info["tables"].values() 
        if info["schema"] == selected_schema
    ]
    
    selected_table = st.selectbox("Select Table", sorted(tables))
    
    if selected_table:
        # Get table details
        table_info = schema_info["tables"][f"{selected_schema}.{selected_table}"]
        
        # Display columns
        st.write("Columns:")
        columns_df = pd.DataFrame(table_info["columns"])
        st.dataframe(columns_df)
        
        # Display sample data
        if st.button("Show Sample Data"):
            result = asyncio.run(
                st.session_state.orchestrator.process({
                    "workflow": "data_analysis",
                    "action": "execute_query",
                    "parameters": {
                        "query": f"SELECT TOP 10 * FROM {selected_schema}.{selected_table}"
                    }
                })
            )
            
            if result.get("success"):
                st.dataframe(result["data"])
            else:
                st.error(f"Failed to get sample data: {result.get('error')}")

def sales_analysis():
    st.header("Sales Analysis")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    
    # Product category selection
    result = asyncio.run(
        st.session_state.orchestrator.process({
            "workflow": "data_analysis",
            "action": "get_product_categories"
        })
    )
    
    if result.get("success"):
        categories = result["data"]
        selected_categories = st.multiselect("Product Categories", categories)
    
    # Analysis options
    generate_summary = st.checkbox("Generate AI Summary", value=True)
    include_charts = st.checkbox("Include Charts", value=True)
    
    if st.button("Generate Analysis"):
        if not selected_categories:
            st.warning("Please select at least one product category")
            return
            
        result = asyncio.run(
            st.session_state.orchestrator.process({
                "workflow": "data_analysis",
                "action": "analyze_sales",
                "parameters": {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "categories": selected_categories,
                    "generate_summary": generate_summary,
                    "include_charts": include_charts
                }
            })
        )
        
        if result.get("success"):
            # Display summary if requested
            if generate_summary and result["data"].get("summary"):
                st.subheader("Analysis Summary")
                st.write(result["data"]["summary"])
            
            # Display charts if requested
            if include_charts and result["data"].get("charts"):
                st.subheader("Sales Charts")
                for chart in result["data"]["charts"]:
                    st.plotly_chart(chart)
            
            # Display data tables
            st.subheader("Sales Data")
            st.dataframe(result["data"]["sales_data"])
        else:
            st.error(f"Failed to generate analysis: {result.get('error')}")

def report_generator():
    st.header("Report Generator")
    
    # Add tabs for different report generation modes
    tab1, tab2 = st.tabs(["🗣️ Natural Language", "🛠️ Advanced Options"])
    
    with tab1:
        st.subheader("Generate Report Using Natural Language")
        
        # Example prompts
        st.markdown("""
        **Example prompts you can try:**
        - Show me sales performance by product category for the last quarter
        - Generate a customer analysis report showing top spenders and their buying patterns
        - Create an inventory status report highlighting items that need restocking
        - Analyze sales trends by region for the past year
        """)
        
        # User prompt input
        user_prompt = st.text_area(
            "Describe the report you want to generate",
            placeholder="E.g., Show me sales performance by product category for the last quarter",
            height=100
        )
        
        # Optional parameters
        with st.expander("Additional Options"):
            include_charts = st.checkbox("Include visualizations", value=True)
            include_insights = st.checkbox("Include AI-generated insights", value=True)
            output_format = st.selectbox(
                "Output Format",
                ["Interactive Dashboard", "PDF Report", "Excel Spreadsheet", "HTML Report"]
            )
        
        if st.button("Generate Report", type="primary"):
            if not user_prompt:
                st.warning("Please provide a description of the report you want to generate.")
                return
                
            try:
                with st.spinner("Generating your report..."):
                    result = asyncio.run(
                        st.session_state.orchestrator.process({
                            "workflow": "report_generation",
                            "action": "generate_report_from_prompt",
                            "parameters": {
                                "prompt": user_prompt,
                                "include_charts": include_charts,
                                "include_insights": include_insights,
                                "output_format": output_format
                            }
                        })
                    )
                    
                    if result.get("success"):
                        # Display report sections
                        if result["data"].get("summary"):
                            st.subheader("Executive Summary")
                            st.write(result["data"]["summary"])
                        
                        if result["data"].get("charts") and include_charts:
                            st.subheader("Visualizations")
                            for chart in result["data"]["charts"]:
                                st.plotly_chart(chart)
                        
                        if result["data"].get("insights") and include_insights:
                            st.subheader("Key Insights")
                            st.write(result["data"]["insights"])
                        
                        if result["data"].get("data_tables"):
                            st.subheader("Detailed Data")
                            for table_name, df in result["data"]["data_tables"].items():
                                st.write(f"**{table_name}**")
                                st.dataframe(df)
                        
                        # Download options
                        if result["data"].get("report_url"):
                            st.success("Report generated successfully!")
                            st.markdown(f"📥 [Download Full Report]({result['data']['report_url']})")
                    else:
                        st.error(f"Failed to generate report: {result.get('error')}")
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        st.subheader("Advanced Report Configuration")
        
        # Report type selection
        report_type = st.selectbox(
            "Report Type",
            ["Sales Analysis", "Customer Insights", "Inventory Status", "Financial Performance"]
        )
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
        
        # Report-specific parameters
        if report_type == "Sales Analysis":
            st.subheader("Sales Analysis Parameters")
            product_categories = st.multiselect(
                "Product Categories",
                ["All"] + asyncio.run(get_product_categories())
            )
            include_customer_details = st.checkbox("Include Customer Details", value=True)
            sales_metrics = st.multiselect(
                "Sales Metrics",
                ["Revenue", "Units Sold", "Profit Margin", "Average Order Value"],
                default=["Revenue", "Units Sold"]
            )
            
        elif report_type == "Customer Insights":
            st.subheader("Customer Analysis Parameters")
            customer_segments = st.multiselect(
                "Customer Segments",
                ["All", "High Value", "Medium Value", "Low Value"],
                default=["All"]
            )
            include_demographics = st.checkbox("Include Demographics", value=True)
            include_behavior = st.checkbox("Include Buying Behavior", value=True)
            
        elif report_type == "Inventory Status":
            st.subheader("Inventory Parameters")
            warehouse_locations = st.multiselect(
                "Warehouse Locations",
                ["All"] + asyncio.run(get_warehouse_locations())
            )
            include_reorder = st.checkbox("Include Reorder Suggestions", value=True)
            include_trends = st.checkbox("Include Historical Trends", value=True)
            
        elif report_type == "Financial Performance":
            st.subheader("Financial Parameters")
            metrics = st.multiselect(
                "Financial Metrics",
                ["Revenue", "Costs", "Profit", "ROI", "Margins"],
                default=["Revenue", "Profit"]
            )
            comparison_period = st.selectbox(
                "Compare With",
                ["Previous Period", "Same Period Last Year", "None"]
            )
        
        # Common options
        st.subheader("Report Options")
        col1, col2 = st.columns(2)
        with col1:
            include_charts = st.checkbox("Include Charts", value=True)
            include_insights = st.checkbox("Include AI Insights", value=True)
        with col2:
            output_format = st.selectbox(
                "Output Format",
                ["Interactive Dashboard", "PDF Report", "Excel Spreadsheet", "HTML Report"]
            )
        
        if st.button("Generate Report", type="primary"):
            try:
                with st.spinner("Generating your report..."):
                    # Build parameters based on report type
                    params = {
                        "report_type": report_type,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "include_charts": include_charts,
                        "include_insights": include_insights,
                        "output_format": output_format
                    }
                    
                    # Add report-specific parameters
                    if report_type == "Sales Analysis":
                        params.update({
                            "product_categories": product_categories,
                            "include_customer_details": include_customer_details,
                            "sales_metrics": sales_metrics
                        })
                    elif report_type == "Customer Insights":
                        params.update({
                            "customer_segments": customer_segments,
                            "include_demographics": include_demographics,
                            "include_behavior": include_behavior
                        })
                    elif report_type == "Inventory Status":
                        params.update({
                            "warehouse_locations": warehouse_locations,
                            "include_reorder": include_reorder,
                            "include_trends": include_trends
                        })
                    elif report_type == "Financial Performance":
                        params.update({
                            "metrics": metrics,
                            "comparison_period": comparison_period
                        })
                    
                    result = asyncio.run(
                        st.session_state.orchestrator.process({
                            "workflow": "report_generation",
                            "action": "generate_report",
                            "parameters": params
                        })
                    )
                    
                    if result.get("success"):
                        # Display report sections
                        if result["data"].get("summary"):
                            st.subheader("Executive Summary")
                            st.write(result["data"]["summary"])
                        
                        if result["data"].get("charts") and include_charts:
                            st.subheader("Visualizations")
                            for chart in result["data"]["charts"]:
                                st.plotly_chart(chart)
                        
                        if result["data"].get("insights") and include_insights:
                            st.subheader("Key Insights")
                            st.write(result["data"]["insights"])
                        
                        if result["data"].get("data_tables"):
                            st.subheader("Detailed Data")
                            for table_name, df in result["data"]["data_tables"].items():
                                st.write(f"**{table_name}**")
                                st.dataframe(df)
                        
                        # Download options
                        if result["data"].get("report_url"):
                            st.success("Report generated successfully!")
                            st.markdown(f"📥 [Download Full Report]({result['data']['report_url']})")
                    else:
                        st.error(f"Failed to generate report: {result.get('error')}")
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

async def get_product_categories():
    """Helper function to get product categories."""
    result = await st.session_state.orchestrator.process({
        "workflow": "data_analysis",
        "action": "get_product_categories"
    })
    return result.get("data", []) if result.get("success") else []

async def get_warehouse_locations():
    """Helper function to get warehouse locations."""
    result = await st.session_state.orchestrator.process({
        "workflow": "data_analysis",
        "action": "get_warehouse_locations"
    })
    return result.get("data", []) if result.get("success") else []

def data_visualization():
    st.header("Data Visualization")
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Visualization Type",
        ["Sales Trends", "Product Performance", "Customer Segments", "Geographic Analysis"]
    )
    
    # Time period selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    
    # Visualization-specific parameters
    if viz_type == "Sales Trends":
        grouping = st.selectbox("Group By", ["Day", "Week", "Month", "Quarter", "Year"])
        include_forecast = st.checkbox("Include Forecast", value=False)
    elif viz_type == "Product Performance":
        metric = st.selectbox("Metric", ["Revenue", "Units Sold", "Profit Margin"])
        top_n = st.slider("Top N Products", 5, 50, 10)
    elif viz_type == "Customer Segments":
        segment_by = st.selectbox("Segment By", ["Purchase Frequency", "Average Order Value", "Customer Lifetime Value"])
        n_segments = st.slider("Number of Segments", 2, 10, 5)
    elif viz_type == "Geographic Analysis":
        region_level = st.selectbox("Region Level", ["Country", "State/Province", "City"])
        metric = st.selectbox("Metric", ["Sales", "Customers", "Orders"])
    
    if st.button("Generate Visualization"):
        result = asyncio.run(
            st.session_state.orchestrator.process({
                "workflow": "data_visualization",
                "action": "create_visualization",
                "parameters": {
                    "viz_type": viz_type,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "options": {
                        "grouping": grouping if viz_type == "Sales Trends" else None,
                        "include_forecast": include_forecast if viz_type == "Sales Trends" else None,
                        "metric": metric if viz_type in ["Product Performance", "Geographic Analysis"] else None,
                        "top_n": top_n if viz_type == "Product Performance" else None,
                        "segment_by": segment_by if viz_type == "Customer Segments" else None,
                        "n_segments": n_segments if viz_type == "Customer Segments" else None,
                        "region_level": region_level if viz_type == "Geographic Analysis" else None
                    }
                }
            })
        )
        
        if result.get("success"):
            st.plotly_chart(result["data"]["chart"])
            
            if result["data"].get("insights"):
                st.subheader("Key Insights")
                st.write(result["data"]["insights"])
        else:
            st.error(f"Failed to generate visualization: {result.get('error')}")

if __name__ == "__main__":
    main() 