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
    """Check required environment variables."""
    # Base required variables
    required_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_DRIVER', 'ANTHROPIC_API_KEY']
    
    # Add authentication variables if not using Windows Authentication
    if os.getenv('DB_TRUSTED_CONNECTION', 'yes').lower() != 'yes':
        required_vars.extend(['DB_USER', 'DB_PASSWORD'])
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# Add after imports
def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "./cache",
        "./cache/insights",
        "./cache/llm",
        "./cache/data",
        "./cache/visualizations",
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

# Initialize session state configuration
def init_session_state():
    if 'config' not in st.session_state:
        st.session_state.config = {
            'database': {
                'type': 'mssql',
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT'),
                'database': os.getenv('DB_NAME'),
                'driver': os.getenv('DB_DRIVER'),
                'trusted_connection': os.getenv('DB_TRUSTED_CONNECTION', 'yes'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'echo': os.getenv('DB_ECHO', 'True').lower() == 'true',
                'cache_dir': './cache/database'
            },
            'api_keys': {
                'anthropic': os.getenv('ANTHROPIC_API_KEY')
            },
            'paths': {
                'output_dir': './reports',
                'template_dir': './templates',
                'cache_dir': './cache',
                'logs_dir': './logs'
            }
        }

# Initialize components
async def init_components():
    try:
        # Load configuration
        config = {
            "database": {
                "type": "mssql",
                "host": os.getenv('DB_HOST'),
                "port": int(os.getenv('DB_PORT', '1433')),
                "database": os.getenv('DB_NAME'),
                "driver": os.getenv('DB_DRIVER'),
                "trusted_connection": os.getenv('DB_TRUSTED_CONNECTION', 'yes'),
                "user": os.getenv('DB_USER'),
                "password": os.getenv('DB_PASSWORD'),
                "echo": os.getenv('DB_ECHO', 'True').lower() == 'true',
                "cache_dir": './cache/database'
            },
            "user_interface": {
                "preferences_dir": "./preferences"
            },
            "report_generator": {
                "config": {
                    "template_dir": "./templates",
                    "output_dir": "./reports"
                },
                "output_dir": "./reports"
            },
            "visualization": {
                "theme": "plotly_white",
                "default_height": 500,
                "default_width": 800,
                "cache_dir": "./cache/visualizations"
            },
            "insight_generator": {
                "llm_manager": {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet-20240229",
                    "api_key": st.session_state.config['api_keys']['anthropic'],
                    "cache_dir": "./cache/llm",
                    "cache": {
                        "enabled": True,
                        "ttl": 3600,
                        "max_size": 1000,
                        "exact_match": True
                    },
                    "models": {
                        "claude-3-sonnet-20240229": {
                    "max_tokens": 1000,
                            "temperature": 0.7,
                            "cost_per_1k_tokens": 0.0
                        }
                    },
                    "monitoring": {
                        "enabled": True,
                        "log_level": "INFO"
                    },
                    "cost_limits": {
                        "daily": 10.0,
                        "monthly": 100.0
                    }
                },
                "cache_dir": "./cache/insights"
            },
            "data_manager": {
                "cache_dir": "./cache/data",
                "batch_size": 1000,
                "max_workers": 4,
                "cache_enabled": True,
                "cache_ttl": 3600,
                "max_retries": 3,
                "data_validation": {
                    "enabled": True,
                    "strict_mode": False
                },
                "preprocessing": {
                    "enabled": True,
                    "handle_missing": True,
                    "handle_outliers": True
                }
            }
        }
        
        logger.info("Creating master orchestrator...")
        output_dir = config["report_generator"]["output_dir"]
        anthropic_api_key = st.session_state.config['api_keys']['anthropic']
        
        # Create orchestrator with required arguments and config
        orchestrator = MasterOrchestratorAgent(
            config=config,
            output_dir=output_dir,
            anthropic_api_key=anthropic_api_key
        )
        
        logger.info("Initializing master orchestrator...")
        try:
        success = await orchestrator.initialize()
        if not success:
                raise RuntimeError("Master orchestrator initialization returned False")
        except Exception as e:
            logger.error(f"Master orchestrator initialization failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize master orchestrator: {str(e)}")
        
        logger.info("Master orchestrator initialized successfully")
        return orchestrator
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        raise

def main():
    # Initialize session state first
    init_session_state()
    
    st.title("AdventureWorks Business Intelligence Dashboard")
    
    # Initialize orchestrator if not already present
    if "orchestrator" not in st.session_state:
        try:
            with st.spinner("Initializing application..."):
        st.session_state.orchestrator = asyncio.run(init_components())
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize application: {error_msg}", exc_info=True)
            st.error(
                f"Failed to initialize application: {error_msg}\n\n"
                "Please check your database configuration and connectivity."
            )
            
            # Display configuration for debugging
            if st.checkbox("Show Configuration"):
                st.code(json.dumps(st.session_state.config, indent=2, default=str))
            return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["Schema Explorer", "Schema Configuration", "Report Generator", "Data Visualization"]
    )
    
    try:
        if page == "Schema Explorer":
            schema_explorer()
        elif page == "Schema Configuration":
            schema_configuration()
        elif page == "Report Generator":
            report_generator()
        elif page == "Data Visualization":
            data_visualization()
    except Exception as e:
        logger.error(f"Error in application: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def schema_explorer():
    st.header("Database Schema Explorer")
    
    # Disable Streamlit's automatic number formatting for this page
    st.write("""
    <style>
    .dataframe td:has(div.stNumberElement) div {
        text-align: right;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
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
        
        schema_info = result.get("data", {})
    
    # Schema selection
        if not schema_info or not isinstance(schema_info.get("tables"), list):
            st.error("No tables found in the database or invalid data format")
            return
            
        # Extract unique schemas from tables
        schemas = sorted(list(set(table["schema"] for table in schema_info["tables"])))
        
        if not schemas:
            st.error("No schemas found in the database")
            return
            
    selected_schema = st.selectbox(
        "Select Schema", 
        schemas,
            key="schema_explorer_schema"
    )
    
    # Display tables in selected schema
    st.subheader(f"Tables in {selected_schema} Schema")
    tables = [
            table["name"] 
            for table in schema_info["tables"] 
            if table.get("schema") == selected_schema
        ]
        
        if not tables:
            st.warning(f"No tables found in schema {selected_schema}")
            return
            
        selected_table = st.selectbox("Select Table", sorted(tables), key="schema_explorer_table")
    except Exception as e:
        st.error(f"Error in schema explorer: {str(e)}")
        st.write("Please check the logs for more details.")
        return
    
    if selected_table:
        try:
            # Get detailed schema information for the selected table
            table_result = asyncio.run(
                st.session_state.orchestrator.process({
                    "workflow": "data_analysis",
                    "action": "get_schema_info",
                    "parameters": {
                        "table_name": selected_table,
                        "schema_name": selected_schema
                    }
                })
            )
            
            if not table_result.get("success"):
                st.error(f"Failed to get table information: {table_result.get('error')}")
                return
            
            table_data = table_result.get("data", {})
            
            # Check if we have valid column data
            columns_data = table_data.get("columns", [])
            if not columns_data:
                # Special handling for ProspectiveBuyer table
                if selected_table == "ProspectiveBuyer":
                    st.warning("ProspectiveBuyer table has special structure. Attempting direct query...")
                    # Try to get sample data directly
                    sample_result = asyncio.run(
                st.session_state.orchestrator.process({
                    "workflow": "data_analysis",
                    "action": "execute_query",
                    "parameters": {
                                "query": f"SELECT TOP 1 * FROM {selected_schema}.{selected_table}"
                            }
                        })
                    )
                    
                    if sample_result.get("success") and sample_result.get("data"):
                        st.write("Sample Data (showing column structure):")
                        # Process the data to ensure numeric values don't have commas
                        clean_data = []
                        for item in sample_result["data"]:
                            clean_item = {}
                            for key, value in item.items():
                                # Convert numeric values directly to string without formatting
                                if isinstance(value, (int, float)) and value % 1 == 0:
                                    clean_item[key] = str(int(value))
                                elif isinstance(value, float):
                                    clean_item[key] = str(value)
                                else:
                                    clean_item[key] = value
                            clean_data.append(clean_item)
                        
                        # Convert to DataFrame and display
                        sample_df = pd.DataFrame(clean_data)
                        st.dataframe(sample_df)
                        
                        # Create columns data from sample
                        columns_data = []
                        for col in sample_df.columns:
                            columns_data.append({
                                "name": col,
                                "data_type": str(sample_df[col].dtype),
                                "max_length": None,
                                "is_nullable": "UNKNOWN", 
                                "is_primary_key": "NO",
                                "is_foreign_key": "NO"
                            })
                    else:
                        st.error(f"Failed to get sample data: {sample_result.get('error')}")
                        return
                else:
                    st.warning("No column information available")
                    return
            
            # Create tabs for different sections
            info_tab, relationships_tab, queries_tab, sample_tab = st.tabs([
                "Column Information",
                "Relationships",
                "SQL Queries",
                "Sample Data"
            ])
            
            with info_tab:
                # Display column information
                columns_df = pd.DataFrame(columns_data)
                display_columns = {
                    "name": "Column Name",
                    "data_type": "Data Type",
                    "max_length": "Max Length",
                    "is_nullable": "Nullable",
                    "is_primary_key": "Primary Key",
                    "is_foreign_key": "Foreign Key"
                }
                available_columns = [col for col in display_columns.keys() if col in columns_df.columns]
                display_df = columns_df[available_columns].rename(
                    columns={col: display_columns[col] for col in available_columns}
                )
                st.dataframe(display_df)
            
            with relationships_tab:
                # Display relationships
                fk_columns = [col for col in columns_data if col.get("is_foreign_key") == "YES"]
                if fk_columns:
                    st.subheader("Foreign Key Relationships")
                    for col in fk_columns:
                        ref = col.get("foreign_key_reference", {})
                        if ref:
                            st.write(f"• {col['name']} → {ref.get('schema', '')}.{ref.get('table', '')}.{ref.get('column', '')}")
                
                # Get and display join patterns
                join_patterns_result = asyncio.run(
                    st.session_state.orchestrator.process({
                        "workflow": "schema_configuration",
                        "action": "get_join_patterns",
                        "parameters": {
                            "table_name": selected_table,
                            "schema_name": selected_schema
                        }
                    })
                )
                
                if join_patterns_result.get("success") and join_patterns_result.get("data", {}).get("join_patterns"):
                    st.subheader("Common Join Patterns")
                    join_patterns = join_patterns_result["data"]["join_patterns"]
                    for pattern in join_patterns:
                        st.write(f"**{pattern['description']}**")
                        query = f"SELECT *\nFROM {selected_schema}.{selected_table}\n"
                        for join in pattern.get("related_tables", []):
                            query += f"{join['join_type']} {join['table']} ON {selected_schema}.{selected_table}.{join['source_column']} = {join['table']}.{join['target_column']}\n"
                        st.code(query, language="sql")
                        st.write("**Table Relationships:**")
                        for join in pattern.get("related_tables", []):
                            st.write(f"• {selected_table} → {join['table']} (via {join['source_column']} = {join['target_column']})")
                        st.write("")
            
            with queries_tab:
                # Display SQL queries
                basic_col, advanced_col = st.columns(2)
                with basic_col:
                    st.subheader("Basic Queries")
                    st.write("**Select all records:**")
                    st.code(f"SELECT *\nFROM {selected_schema}.{selected_table};", language="sql")
                    if columns_data:
                        first_column = columns_data[0]["name"]
                        st.write("**Select with WHERE clause:**")
                        st.code(f"SELECT *\nFROM {selected_schema}.{selected_table}\nWHERE {first_column} = @value;", language="sql")
                        st.write("**Select with ORDER BY:**")
                        st.code(f"SELECT *\nFROM {selected_schema}.{selected_table}\nORDER BY {first_column} DESC;", language="sql")
                    st.write("**Count records:**")
                    st.code(f"SELECT COUNT(*)\nFROM {selected_schema}.{selected_table};", language="sql")
                
                with advanced_col:
                    st.subheader("Advanced Queries")
                    numeric_columns = [col["name"] for col in columns_data 
                                    if col["data_type"].upper() in ("INT", "DECIMAL", "NUMERIC", "FLOAT", "MONEY")]
                    if numeric_columns:
                        st.write("**Aggregate with GROUP BY:**")
                        agg_col = numeric_columns[0]
                        group_col = next((col["name"] for col in columns_data 
                                        if col["name"] not in numeric_columns), first_column)
                        st.code(f"""SELECT {group_col},
       COUNT(*) as count,
       SUM({agg_col}) as total_{agg_col},
       AVG({agg_col}) as avg_{agg_col}
FROM {selected_schema}.{selected_table}
GROUP BY {group_col}
ORDER BY count DESC;""", language="sql")
                    
                    st.write("**Query with Subquery:**")
                    st.code(f"""SELECT *
FROM {selected_schema}.{selected_table} t1
WHERE {first_column} IN (
    SELECT {first_column}
    FROM {selected_schema}.{selected_table} t2
    WHERE /* your condition here */
);""", language="sql")
            
            with sample_tab:
                if st.button("Load Sample Data"):
                    sample_result = asyncio.run(
                st.session_state.orchestrator.process({
                    "workflow": "data_analysis",
                            "action": "execute_query",
                    "parameters": {
                                "query": f"SELECT TOP 10 * FROM {selected_schema}.{selected_table}"
                            }
                        })
                    )
                    
                    if sample_result.get("success") and sample_result.get("data"):
                        clean_data = []
                        for item in sample_result["data"]:
                            clean_item = {}
                            for key, value in item.items():
                                if isinstance(value, (int, float)) and value % 1 == 0:
                                    clean_item[key] = str(int(value))
                                elif isinstance(value, float):
                                    clean_item[key] = str(value)
            else:
                                    clean_item[key] = value
                            clean_data.append(clean_item)
                        
                        sample_df = pd.DataFrame(clean_data)
                        st.dataframe(sample_df)
            else:
                        st.error(f"Failed to get sample data: {sample_result.get('error')}")
        except Exception as e:
            st.error(f"Error displaying table information: {str(e)}")
            st.write("Please check the logs for more details.")

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
                ["Interactive Dashboard", "PDF Report", "Excel Spreadsheet", "HTML Report"],
                key="nl_output_format"
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
            ["Sales Analysis", "Customer Insights", "Inventory Status", "Financial Performance"],
            key="advanced_report_type"
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
                key="sales_product_categories"
            )
            include_customer_details = st.checkbox("Include Customer Details", value=True, key="sales_customer_details")
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
            include_demographics = st.checkbox("Include Demographics", value=True, key="customer_demographics")
            include_behavior = st.checkbox("Include Buying Behavior", value=True, key="customer_behavior")
            
        elif report_type == "Inventory Status":
            st.subheader("Inventory Parameters")
            warehouse_locations = st.multiselect(
                "Warehouse Locations",
                ["All"] + asyncio.run(get_warehouse_locations()),
                key="inventory_locations"
            )
            include_reorder = st.checkbox("Include Reorder Suggestions", value=True, key="inventory_reorder")
            include_trends = st.checkbox("Include Historical Trends", value=True, key="inventory_trends")
            
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
                key="financial_comparison"
            )
        
        # Common options
        st.subheader("Report Options")
        col1, col2 = st.columns(2)
        with col1:
            include_charts = st.checkbox("Include Charts", value=True, key="advanced_charts")
            include_insights = st.checkbox("Include AI Insights", value=True, key="advanced_insights")
        with col2:
            output_format = st.selectbox(
                "Output Format",
                ["Interactive Dashboard", "PDF Report", "Excel Spreadsheet", "HTML Report"],
                key="advanced_output_format"
            )
        
        if st.button("Generate Report", type="primary", key="advanced_generate"):
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

def schema_configuration():
    st.header("Schema Configuration")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Table Categories", "Suggested Joins", "SQL Examples"])
    
    with tab1:
        with st.expander("Table Categories", expanded=False):
            try:
                # Get schema information through orchestrator
                schema_info = asyncio.run(
                    st.session_state.orchestrator.process({
                        "workflow": "data_analysis",
                        "action": "get_schema_info"
                    })
                )
                
                if schema_info.get("success"):
                    tables = schema_info["data"]["tables"]
                    
                    # Group tables by category
                    categories = {}
                    for table in tables:
                        # Use the first part of the table name as category
                        category = table["name"].split("_")[0] if "_" in table["name"] else "Other"
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(table)
                    
                    # Display tables by category
                    for category, tables in categories.items():
                        st.subheader(f"{category} Tables")
                        for table in tables:
                            st.write(f"- {table['name']} ({table['column_count']} columns)")
                else:
                    st.error("Failed to get schema information")
            except Exception as e:
                st.error(f"Error in schema configuration: {str(e)}")
    
    with tab2:
        with st.expander("Suggested Joins", expanded=False):
            try:
                # Get schema information first to get list of tables
                schema_info = asyncio.run(
                    st.session_state.orchestrator.process({
                        "workflow": "data_analysis",
                        "action": "get_schema_info"
                    })
                )
                
                if not schema_info.get("success"):
                    st.error("Failed to get schema information")
                    return
                
                tables = schema_info["data"]["tables"]
                all_join_patterns = []
                
                # Get join patterns for each table
                for table in tables:
                    join_patterns = asyncio.run(
                        st.session_state.orchestrator.process({
                            "workflow": "schema_configuration",
                            "action": "get_join_patterns",
                            "parameters": {
                                "table_name": table["name"],
                                "schema_name": table["schema"]
                            }
                        })
                    )
                    
                    if join_patterns.get("success"):
                        all_join_patterns.extend(join_patterns["data"]["join_patterns"])
                
                # Display join patterns
                if all_join_patterns:
                    for pattern in all_join_patterns:
                        st.subheader(pattern["description"])
                        for table in pattern["related_tables"]:
                            st.write(f"- Join with {table['table']} using {table['source_column']} = {table['target_column']}")
                else:
                    st.info("No join patterns found in the database")
    except Exception as e:
                st.error(f"Error getting join patterns: {str(e)}")
    
    with tab3:
        with st.expander("SQL Examples", expanded=False):
            try:
                # Basic Queries Tab
                basic_tab, advanced_tab = st.tabs(["Basic Queries", "Advanced Queries"])
                
                with basic_tab:
                    st.subheader("Basic SQL Query Examples")
                    st.code("""
-- Example 1: Simple SELECT
SELECT column1, column2
FROM table_name
WHERE condition;

-- Example 2: JOIN
SELECT t1.column1, t2.column2
FROM table1 t1
JOIN table2 t2 ON t1.id = t2.id;

-- Example 3: Aggregation
SELECT column1, COUNT(*) as count
FROM table_name
GROUP BY column1;
                    """)
                
                with advanced_tab:
                    st.subheader("Advanced SQL Query Examples")
                    st.code("""
-- Example 1: Window Functions
SELECT column1,
       ROW_NUMBER() OVER (PARTITION BY column2 ORDER BY column3) as row_num
FROM table_name;

-- Example 2: Common Table Expression (CTE)
WITH cte_name AS (
    SELECT column1, column2
    FROM table_name
)
SELECT * FROM cte_name;

-- Example 3: Pivot
SELECT *
FROM (SELECT column1, column2, value
      FROM table_name) p
PIVOT (MAX(value) FOR column2 IN ('A', 'B', 'C')) as pvt;
                    """)
            except Exception as e:
                st.error(f"Error displaying SQL examples: {str(e)}")

if __name__ == "__main__":
    main() 