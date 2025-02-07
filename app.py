import streamlit as st
import json
import logging
from pathlib import Path
from datetime import datetime
from agents.master_orchestrator_agent import MasterOrchestratorAgent
import asyncio
import pandas as pd
from agents import initialize_environment
import os
from utils.schema_configurator import SchemaConfigurator

# Add this after initialize_environment()
def check_environment():
    required_vars = [
        'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 
        'DB_DRIVER', 'OPENAI_API_KEY'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# Add after imports
def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "./cache",
        "./cache/insights",
        "./preferences",
        "./reports",
        "./templates",
        "./logs"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Add before initialize_environment()
ensure_directories()

# Initialize environment and check variables
initialize_environment()
check_environment()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
async def init_components():
    try:
        # Load configuration
        config = {
            "database": st.session_state.config['database'],
            "user_interface": {
                "preferences_dir": "./preferences"
            },
            "report_generator": {
                "config": {
                    "template_dir": "./templates",
                    "output_dir": "./reports"
                },
                "output_dir": "./reports",  # Direct parameters for constructor
                "openai_api_key": st.session_state.config['api_keys']['openai']  # Direct parameter for constructor
            },
            "insight_generator": {
                "llm_manager": {
                    "provider": "openai",
                    "model": "gpt-4-turbo-preview",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "api_key": st.session_state.config['api_keys']['openai']
                },
                "cache_dir": "./cache/insights"
            },
            "data_manager": {
                "cache_dir": "./cache"
            }
        }
        
        logger.info("Creating master orchestrator...")
        orchestrator = MasterOrchestratorAgent(config)
        
        logger.info("Initializing master orchestrator...")
        success = await orchestrator.initialize()
        
        if not success:
            logger.error("Failed to initialize master orchestrator")
            raise RuntimeError("Failed to initialize master orchestrator")
        
        logger.info("Master orchestrator initialized successfully")
        return orchestrator
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
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
        ["Schema Explorer", "Schema Configuration", "Sales Analysis", "Report Generator", "Data Visualization"]
    )
    
    try:
        if page == "Schema Explorer":
            schema_explorer()
        elif page == "Schema Configuration":
            schema_configuration()
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
    selected_schema = st.selectbox(
        "Select Schema", 
        schemas,
        key="schema_selector"
    )
    
    # Display tables in selected schema
    st.subheader(f"Tables in {selected_schema} Schema")
    tables = [
        info["name"] 
        for info in schema_info["tables"].values() 
        if info["schema"] == selected_schema
    ]
    
    selected_table = st.selectbox(
        "Select Table", 
        sorted(tables),
        key="table_selector"
    )
    
    if selected_table:
        # Get table details
        full_table_name = f"{selected_schema}.{selected_table}"
        table_info = schema_info["tables"][full_table_name]
        
        # Display table structure
        st.subheader("Table Structure")
        columns_df = pd.DataFrame(table_info["columns"])
        st.dataframe(columns_df)
        
        # Display foreign keys if any
        if table_info.get("foreign_keys"):
            st.subheader("Foreign Keys")
            fk_df = pd.DataFrame(table_info["foreign_keys"])
            st.dataframe(fk_df)
        
        # Display sample data
        if st.button("Show Sample Data", key="show_sample_data_btn"):
            result = asyncio.run(
                st.session_state.orchestrator.process({
                    "workflow": "data_analysis",
                    "action": "get_sample_data",
                    "parameters": {
                        "table_name": full_table_name,
                        "limit": 10
                    }
                })
            )
            
            if result.get("success"):
                st.subheader("Sample Data")
                st.dataframe(result["data"])
                with st.expander("Debug Logs"):
                    st.text(result.get("logs", "No logs available"))
            else:
                st.error(f"Failed to get sample data: {result.get('error')}")
                with st.expander("Debug Information"):
                    st.text(result.get("logs", "No logs available"))

def sales_analysis():
    st.header("Sales Analysis")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", key="sales_start_date")
    with col2:
        end_date = st.date_input("End Date", key="sales_end_date")
    
    # Product category selection
    result = asyncio.run(
        st.session_state.orchestrator.process({
            "workflow": "data_analysis",
            "action": "get_product_categories"
        })
    )
    
    if result.get("success"):
        categories = result["data"]
        selected_categories = st.multiselect(
            "Product Categories", 
            categories,
            key="product_categories"  # Add unique key
        )
    
    # Analysis options
    generate_summary = st.checkbox("Generate AI Summary", value=True)
    include_charts = st.checkbox("Include Charts", value=True)
    
    if st.button("Generate Analysis", key="generate_analysis_btn"):
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
            height=100,
            key="report_prompt"  # Add unique key
        )
        
        # Optional parameters
        with st.expander("Additional Options"):
            include_charts = st.checkbox("Include visualizations", value=True, key="include_viz")
            include_insights = st.checkbox("Include AI-generated insights", value=True, key="include_insights")
            output_format = st.selectbox(
                "Output Format",
                ["Interactive Dashboard", "PDF Report", "Excel Spreadsheet", "HTML Report"],
                key="output_format"  # Add unique key
            )
        
        if st.button("Generate Report", type="primary", key="nl_generate_report_btn"):
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
            ["Sales Analysis", "Customer Insights", "Inventory Status", "Financial Performance"],
            key="advanced_report_type"  # Add unique key
        )
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", key="advanced_start_date")
        with col2:
            end_date = st.date_input("End Date", key="advanced_end_date")
        
        # Report-specific parameters
        if report_type == "Sales Analysis":
            st.subheader("Sales Analysis Parameters")
            product_categories = st.multiselect(
                "Product Categories",
                ["All"] + asyncio.run(get_product_categories()),
                key="adv_product_categories"
            )
            include_customer_details = st.checkbox(
                "Include Customer Details", 
                value=True,
                key="include_customer_details"
            )
            sales_metrics = st.multiselect(
                "Sales Metrics",
                ["Revenue", "Units Sold", "Profit Margin", "Average Order Value"],
                default=["Revenue", "Units Sold"],
                key="sales_metrics"
            )
            
        elif report_type == "Customer Insights":
            st.subheader("Customer Analysis Parameters")
            customer_segments = st.multiselect(
                "Customer Segments",
                ["All", "High Value", "Medium Value", "Low Value"],
                default=["All"],
                key="customer_segments"
            )
            include_demographics = st.checkbox(
                "Include Demographics", 
                value=True,
                key="include_demographics"
            )
            include_behavior = st.checkbox(
                "Include Buying Behavior", 
                value=True,
                key="include_behavior"
            )
            
        elif report_type == "Inventory Status":
            st.subheader("Inventory Parameters")
            warehouse_locations = st.multiselect(
                "Warehouse Locations",
                ["All"] + asyncio.run(get_warehouse_locations()),
                key="warehouse_locations"
            )
            include_reorder = st.checkbox(
                "Include Reorder Suggestions", 
                value=True,
                key="include_reorder"
            )
            include_trends = st.checkbox(
                "Include Historical Trends", 
                value=True,
                key="include_trends"
            )
            
        elif report_type == "Financial Performance":
            st.subheader("Financial Parameters")
            metrics = st.multiselect(
                "Financial Metrics",
                ["Revenue", "Costs", "Profit", "ROI", "Margins"],
                default=["Revenue", "Profit"],
                key="financial_metrics"
            )
            comparison_period = st.selectbox(
                "Compare With",
                ["Previous Period", "Same Period Last Year", "None"],
                key="comparison_period"
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
                ["Interactive Dashboard", "PDF Report", "Excel Spreadsheet", "HTML Report"],
                key="adv_output_format"
            )
        
        if st.button("Generate Report", type="primary", key="adv_generate_report_btn"):
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
        ["Sales Trends", "Product Performance", "Customer Segments", "Geographic Analysis"],
        key="viz_type"  # Add unique key
    )
    
    # Time period selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", key="viz_start_date")
    with col2:
        end_date = st.date_input("End Date", key="viz_end_date")
    
    # Visualization-specific parameters
    if viz_type == "Sales Trends":
        grouping = st.selectbox(
            "Group By", 
            ["Day", "Week", "Month", "Quarter", "Year"],
            key="sales_grouping"  # Add unique key
        )
        include_forecast = st.checkbox("Include Forecast", value=False, key="include_forecast")
    elif viz_type == "Product Performance":
        metric = st.selectbox(
            "Metric", 
            ["Revenue", "Units Sold", "Profit Margin"],
            key="product_metric"  # Add unique key
        )
        top_n = st.slider("Top N Products", 5, 50, 10, key="top_n_products")
    elif viz_type == "Customer Segments":
        segment_by = st.selectbox(
            "Segment By", 
            ["Purchase Frequency", "Average Order Value", "Customer Lifetime Value"],
            key="segment_by"
        )
        n_segments = st.slider("Number of Segments", 2, 10, 5)
    elif viz_type == "Geographic Analysis":
        region_level = st.selectbox(
            "Region Level", 
            ["Country", "State/Province", "City"],
            key="region_level"
        )
        metric = st.selectbox(
            "Metric", 
            ["Sales", "Customers", "Orders"],
            key="geo_metric"
        )
    
    if st.button("Generate Visualization", key="generate_viz_btn"):
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

def schema_configuration():
    st.header("Schema Configuration")
    
    if "schema_configurator" not in st.session_state:
        st.session_state.schema_configurator = SchemaConfigurator(
            st.session_state.orchestrator.agents["database"]
        )
    
    try:
        config = asyncio.run(st.session_state.schema_configurator.configure_schema())
        
        # Update session state config
        if config:
            st.session_state.config['database']['schema_mapping'] = config
            st.success("Schema configuration updated successfully!")
            
    except Exception as e:
        st.error(f"Error configuring schema: {str(e)}")

if __name__ == "__main__":
    main() 