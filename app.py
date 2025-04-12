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
import plotly.graph_objects as go

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
                'host': os.getenv('DB_HOST', ''),
                'port': os.getenv('DB_PORT', ''),
                'database': os.getenv('DB_NAME', ''),
                'driver': os.getenv('DB_DRIVER', ''),
                'trusted_connection': os.getenv('DB_TRUSTED_CONNECTION', 'yes'),
                'user': os.getenv('DB_USER', ''),
                'password': os.getenv('DB_PASSWORD', ''),
                'echo': os.getenv('DB_ECHO', 'True').lower() == 'true',
                'cache_dir': './cache/database'
            },
            'api_keys': {
                'anthropic': os.getenv('ANTHROPIC_API_KEY', '')
            },
            'paths': {
                'output_dir': './reports',
                'template_dir': './templates',
                'cache_dir': './cache',
                'logs_dir': './logs'
            },
            # Add missing agent configurations
            'data_manager': {
                'cache_dir': './cache/data'
            },
            'user_interface': {
                'theme': 'light'
            },
            'visualization': {
                'theme': 'streamlit',
                'cache_dir': './cache/visualizations'
            },
            'report_generator': {
                'config': {
                    'templates_dir': './templates',
                    'report_format': ['html', 'pdf']
                },
                'output_dir': './reports'
            },
            'insight_generator': {
                'cache_dir': './cache/insights',
                'model': 'claude-3-sonnet-20240229',
                'llm_manager': {
                    'provider': 'anthropic',
                    'model': 'claude-3.7-sonnet-20240620',
                    'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
                    'cache_dir': './cache/llm',
                    'temperature': 0.7,
                    'max_tokens': 4000
                }
            },
            'assistant': {
                'model': 'claude-3-sonnet-20240229'
            },
            'llm_manager': {
                'provider': 'anthropic',
                'model': 'claude-3.7-sonnet-20240620',
                'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
                'cache_dir': './cache/llm',
                'temperature': 0.7,
                'max_tokens': 4000
            }
        }

# Initialize components
async def init_components():
    """Initialize application components."""
    try:
        # Log config for debugging
        logger.info(f"Initializing with config: {st.session_state.config}")
        
        # Make sure all required sections are in the config
        ensure_config_structure()
        
        # Initialize orchestrator with all required parameters
        orchestrator = MasterOrchestratorAgent(
            config=st.session_state.config,
            output_dir=st.session_state.config.get("paths", {}).get("output_dir", "./reports"),
            anthropic_api_key=st.session_state.config.get("api_keys", {}).get("anthropic", "")
        )
        success = await orchestrator.initialize()
        
        if not success:
            st.error("Failed to initialize orchestrator")
            return False
            
        # Store orchestrator in session state
        st.session_state.orchestrator = orchestrator
        
        # Initialize other components
        st.session_state.report_history = []
        st.session_state.current_report = None
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        st.error(f"Error initializing components: {str(e)}")
        return False

def ensure_config_structure():
    """Ensure all required sections are in the config."""
    # Check and add paths if missing
    if "paths" not in st.session_state.config:
        st.session_state.config["paths"] = {
            "output_dir": "./reports",
            "template_dir": "./templates",
            "cache_dir": "./cache",
            "logs_dir": "./logs"
        }
    
    # Check and add api_keys if missing
    if "api_keys" not in st.session_state.config:
        st.session_state.config["api_keys"] = {
            "anthropic": os.getenv("ANTHROPIC_API_KEY", "")
        }
    
    # Add LLM manager configuration
    if "llm_manager" not in st.session_state.config:
        st.session_state.config["llm_manager"] = {
            "provider": "anthropic",
            "model": "claude-3.7-sonnet-20240620",
            "api_key": st.session_state.config.get("api_keys", {}).get("anthropic", ""),
            "cache_dir": "./cache/llm",
            "temperature": 0.7,
            "max_tokens": 4000
        }
    
    # Required agent config sections
    required_sections = [
        "database", "data_manager", "user_interface", "visualization", 
        "report_generator", "insight_generator", "assistant"
    ]
    
    # Add missing sections with minimal config
    for section in required_sections:
        if section not in st.session_state.config:
            if section == "database":
                st.session_state.config[section] = {
                    'type': 'mssql',
                    'host': os.getenv('DB_HOST', ''),
                    'port': os.getenv('DB_PORT', ''),
                    'database': os.getenv('DB_NAME', ''),
                    'driver': os.getenv('DB_DRIVER', ''),
                    'trusted_connection': os.getenv('DB_TRUSTED_CONNECTION', 'yes'),
                    'user': os.getenv('DB_USER', ''),
                    'password': os.getenv('DB_PASSWORD', ''),
                    'echo': os.getenv('DB_ECHO', 'True').lower() == 'true',
                    'cache_dir': './cache/database'
                }
            elif section == "report_generator":
                st.session_state.config[section] = {
                    'config': {
                        'templates_dir': './templates',
                        'report_format': ['html', 'pdf']
                    },
                    'output_dir': './reports'
                }
            elif section == "insight_generator":
                st.session_state.config[section] = {
                    'cache_dir': './cache/insights',
                    'llm_manager': {
                        'provider': 'anthropic',
                        'model': 'claude-3.7-sonnet-20240620',
                        'api_key': st.session_state.config.get("api_keys", {}).get("anthropic", ""),
                        'cache_dir': './cache/llm',
                        'temperature': 0.7,
                        'max_tokens': 4000
                    }
                }
            else:
                st.session_state.config[section] = {
                    'cache_dir': f'./cache/{section}'
                }
    
    # Make sure insight_generator has llm_manager config
    if "insight_generator" in st.session_state.config and "llm_manager" not in st.session_state.config["insight_generator"]:
        st.session_state.config["insight_generator"]["llm_manager"] = {
            'provider': 'anthropic',
            'model': 'claude-3.7-sonnet-20240620',
            'api_key': st.session_state.config.get("api_keys", {}).get("anthropic", ""),
            'cache_dir': './cache/llm',
            'temperature': 0.7,
            'max_tokens': 4000
        }

def main():
    # Initialize session state first
    init_session_state()
    
    st.title("Business Intelligence Dashboard")
    
    # Initialize orchestrator if not already present
    if "orchestrator" not in st.session_state:
        try:
            with st.spinner("Initializing application..."):
                # Create and run the async initialization
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(init_components())
                loop.close()
                
                if not success:
                    st.error("Failed to initialize application components")
                    
                    # Display configuration for debugging
                    if st.checkbox("Show Configuration"):
                        st.code(json.dumps(st.session_state.config, indent=2, default=str))
                    return
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
        ["Schema Explorer", "Schema Configuration", "Report Generator"]
    )
    
    try:
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if page == "Schema Explorer":
            loop.run_until_complete(schema_explorer())
        elif page == "Schema Configuration":
            loop.run_until_complete(schema_configuration())
        elif page == "Report Generator":
            loop.run_until_complete(report_generator())
            
        # Close the event loop
        loop.close()
    except Exception as e:
        logger.error(f"Error in application: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

        # Allow viewing configuration for debugging
        if st.checkbox("Show Debug Information"):
            st.subheader("Configuration")
            st.code(json.dumps(st.session_state.config, indent=2, default=str))
            
            if "orchestrator" in st.session_state:
                st.subheader("Orchestrator Status")
                st.write(f"Orchestrator initialized: {st.session_state.orchestrator is not None}")
                if st.session_state.orchestrator:
                    st.write(f"Active agents: {list(st.session_state.orchestrator.agents.keys())}")
            else:
                st.write("Orchestrator not initialized.")

async def schema_explorer():
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
        result = await st.session_state.orchestrator.process({
            "workflow": "data_analysis",
            "action": "get_schema_info"
        })
        
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
        
        if selected_table:
            # Get detailed schema information for the selected table
            table_result = await st.session_state.orchestrator.process({
                "workflow": "data_analysis",
                "action": "get_schema_info",
                "parameters": {
                    "table_name": selected_table,
                    "schema_name": selected_schema
                }
            })
            
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
                    sample_result = await st.session_state.orchestrator.process({
                        "workflow": "data_analysis",
                        "action": "execute_query",
                        "parameters": {
                            "query": f"SELECT TOP 1 * FROM {selected_schema}.{selected_table}"
                        }
                    })
                    
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
                        df = pd.DataFrame(clean_data)
                        st.dataframe(df)
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
                join_patterns_result = await st.session_state.orchestrator.process({
                    "workflow": "schema_configuration",
                    "action": "get_join_patterns",
                    "parameters": {
                        "table_name": selected_table,
                        "schema_name": selected_schema
                    }
                })
                
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
                    sample_result = await st.session_state.orchestrator.process({
                        "workflow": "data_analysis",
                        "action": "execute_query",
                        "parameters": {
                            "query": f"SELECT TOP 10 * FROM {selected_schema}.{selected_table}"
                        }
                    })
                    
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
        st.error(f"Error in schema explorer: {str(e)}")
        st.write("Please check the logs for more details.")
        return

async def report_generator():
    st.header("Report Generator")
    
    # Add tabs for different report generation modes
    tab1, tab2 = st.tabs(["🗣️ Natural Language", "🛠️ Advanced Options"])
    
    with tab1:
        st.subheader("Generate Report Using Natural Language")
        
        # Example prompts
        st.markdown("""
        **Example prompts you can try:**
        - Show me performance metrics by category for the last quarter
        - Generate an analysis report showing top items and their patterns
        - Create a status report highlighting items that need attention
        - Analyze trends by region for the past year
        """)
        
        # User prompt input
        user_prompt = st.text_area(
            "Describe the report you want to generate",
            placeholder="E.g., Show me performance metrics by category for the last quarter",
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
                    result = await st.session_state.orchestrator.process({
                            "workflow": "report_generation",
                            "action": "generate_report_from_prompt",
                            "parameters": {
                                "prompt": user_prompt,
                                "include_charts": include_charts,
                                "include_insights": include_insights,
                                "output_format": output_format
                            }
                        })
                    
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
        
        # Get the available tables for the user to choose from
        schema_info = await st.session_state.orchestrator.process({
            "workflow": "data_analysis",
            "action": "get_schema_info"
        })
        
        if not schema_info.get("success"):
            st.error("Failed to get schema information")
            return
            
        tables = schema_info.get("data", {}).get("tables", [])
        if not tables:
            st.warning("No tables found in the database.")
            return
        table_options = [f"{table['schema']}.{table['name']}" for table in tables]
        
        # --- Data Selection --- 
        st.markdown("#### Data Selection")
        primary_table = st.selectbox(
            "Primary Table",
            table_options,
            key="primary_table",
            index=0 # Default to the first table
        )
        
        # Fields to include (dynamic based on selected table)
        dimensions = []
        metrics = []
        date_columns = []
        all_columns = [] 
        if primary_table:
            schema_name, table_name = primary_table.split(".")
            table_info = await st.session_state.orchestrator.process({
                "workflow": "data_analysis",
                "action": "get_schema_info",
                "parameters": {"table_name": table_name, "schema_name": schema_name}
            })
            
            if table_info.get("success") and table_info.get("data", {}).get("columns"):
                columns = table_info["data"]["columns"]
                all_columns = [col["name"] for col in columns]
                text_columns = [col["name"] for col in columns if col["data_type"].lower() in ("varchar", "nvarchar", "char", "text", "nchar", "string")]
                numeric_columns = [col["name"] for col in columns if col["data_type"].lower() in ("int", "integer", "bigint", "smallint", "tinyint", "decimal", "numeric", "float", "real", "double", "money", "smallmoney")]
                date_columns = [col["name"] for col in columns if col["data_type"].lower() in ("date", "datetime", "datetime2", "smalldatetime", "timestamp")]
                
                dimensions = st.multiselect(
                    "Dimensions (Group By)",
                    text_columns + date_columns,
                    key="dimensions",
                    help="Select columns to group or categorize data (e.g., Product Name, Region, Date)."
                )
                
                metrics = st.multiselect(
                    "Metrics (Aggregate)",
                    numeric_columns,
                    key="metrics",
                    help="Select numerical columns to measure or aggregate (e.g., Sales Amount, Quantity)."
                )
            else:
                 st.warning("Could not retrieve column information for the selected table.")

        # --- Time Period --- 
        st.markdown("#### Time Period (Optional)")
        if date_columns: # Only show if date columns are available
            time_column = st.selectbox("Time Column", ["None"] + date_columns, key="time_column")
            if time_column != "None":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", key="advanced_start_date")
                with col2:
                    end_date = st.date_input("End Date", key="advanced_end_date")
            else:
                start_date = None
                end_date = None
        else:
            st.info("No date/time columns detected in the selected table for time period filtering.")
            time_column = None
            start_date = None
            end_date = None

        # --- Filtering --- 
        st.markdown("#### Filters (Optional)")
        with st.expander("Add Filters"):
            if all_columns:
                filter_columns = st.multiselect(
                    "Filter Columns",
                    all_columns,
                    key="filter_columns"
                )
                
                filters = {}
                # Need full column info again here for type checking
                # This is slightly inefficient, consider caching table_info if performance is an issue
                if filter_columns and table_info.get("success"):
                    columns_detailed = table_info["data"]["columns"]
                    for col_name in filter_columns:
                        col_info = next((c for c in columns_detailed if c["name"] == col_name), None)
                        if col_info:
                            col_type = col_info["data_type"].lower()
                            if col_type in ("varchar", "nvarchar", "char", "text", "nchar", "string"):
                                filters[col_name] = st.text_input(f"Filter for {col_name} (text)", key=f"filter_{col_name}")
                            elif col_type in ("int", "integer", "bigint", "smallint", "tinyint", "decimal", "numeric", "float", "real", "double", "money", "smallmoney"):
                                # Simple equals for now, range could be added
                                filters[col_name] = st.number_input(f"Filter for {col_name} (number)", key=f"filter_{col_name}")
                            elif col_type in ("date", "datetime", "datetime2", "smalldatetime", "timestamp"):
                                filter_date = st.date_input(f"Date for {col_name}", key=f"filter_{col_name}")
                                filters[col_name] = filter_date.strftime("%Y-%m-%d")
            else:
                st.write("Select a primary table to enable filters.")
                filters = {}

        # --- Visualization Selection --- 
        st.markdown("#### Visualizations (Optional)")
        include_charts = st.checkbox("Include Charts in Report", value=True, key="advanced_charts")
        chart_types = []
        if include_charts:
            chart_options = ["Auto", "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Table"] # Add more as needed
            chart_types = st.multiselect(
                "Select Chart Types",
                chart_options,
                default=["Auto"],
                key="chart_types",
                help="Select specific charts. 'Auto' tries to generate relevant charts based on data."
            )

        # --- Other Report Options --- 
        st.markdown("#### Report Options")
        include_insights = st.checkbox("Include AI Insights", value=True, key="advanced_insights")
        output_format = st.selectbox(
            "Output Format",
            ["Interactive Dashboard", "PDF Report", "Excel Spreadsheet", "HTML Report"],
            key="advanced_output_format"
        )
        
        # --- Generate Button --- 
        if st.button("Generate Advanced Report", type="primary", key="advanced_generate"):
            # Basic validation
            if not primary_table:
                st.warning("Please select a Primary Table.")
                return
            if not metrics:
                st.warning("Please select at least one Metric.")
                return

            try:
                with st.spinner("Generating your report..."):
                    # Build parameters for the backend
                    params = {
                        "primary_table": primary_table,
                        "dimensions": dimensions,
                        "metrics": metrics,
                        "filters": filters,
                        "include_charts": include_charts,
                        "chart_types": chart_types, # Pass selected chart types
                        "include_insights": include_insights,
                        "output_format": output_format
                    }
                    
                    # Add time period if selected
                    if time_column and time_column != "None" and start_date and end_date:
                        params.update({
                            "time_column": time_column,
                            "start_date": start_date.strftime("%Y-%m-%d"),
                            "end_date": end_date.strftime("%Y-%m-%d"),
                        })
                    
                    # Call orchestrator with generate_report action
                    result = await st.session_state.orchestrator.process({
                        "workflow": "report_generation",
                        "action": "generate_report", # Use the structured action
                        "parameters": params
                    })
                    
                    # Display results (same logic as before)
                    if result.get("success"):
                        report_data = result.get("data", {})
                        if report_data.get("summary"):
                            st.subheader("Executive Summary")
                            st.write(report_data["summary"])
                        
                        if report_data.get("charts"):
                            st.subheader("Visualizations")
                            for chart in report_data["charts"]:
                                # Ensure chart is a Plotly figure
                                if isinstance(chart, go.Figure):
                                     st.plotly_chart(chart)
                                else:
                                     logger.warning(f"Item in charts list is not a Plotly Figure: {type(chart)}")
                        
                        if report_data.get("insights"):
                            st.subheader("Key Insights")
                            st.write(report_data["insights"])
                        
                        if report_data.get("data_tables"):
                            st.subheader("Detailed Data")
                            for table_name, df_data in report_data["data_tables"].items():
                                st.write(f"**{table_name}**")
                                # Handle potential non-DataFrame data
                                if isinstance(df_data, pd.DataFrame):
                                     st.dataframe(df_data)
                                else:
                                     st.write("Data is not in a table format.")
                                     st.code(str(df_data))
                        
                        if report_data.get("report_url"):
                            st.success("Report generated successfully!")
                            st.markdown(f"📥 [Download Full Report]({report_data['report_url']})")
                        elif not any(report_data.get(key) for key in ["summary", "charts", "insights", "data_tables"]):
                            st.info("Report generated, but no specific content (summary, charts, insights, data) was produced.")
                            if report_data.get("generated_sql"):
                                st.code(f"Generated SQL:\n{report_data['generated_sql']}", language="sql")

                    else:
                        st.error(f"Failed to generate report: {result.get('error', 'Unknown error')}")
                        # Optionally show generated SQL if available in error
                        if result.get('data', {}).get("generated_sql"):
                             st.code(f"Generated SQL (failed):\n{result['data']['generated_sql']}", language="sql")
                        
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.error(f"UI error during advanced report generation: {e}", exc_info=True)

# Remove sales_analysis function and implement more generic helper functions

async def get_table_hierarchies():
    """Helper function to get table hierarchies (parent-child relationships)."""
    result = await st.session_state.orchestrator.process({
        "workflow": "schema_configuration",
        "action": "get_table_hierarchies"
    })
    return result.get("data", []) if result.get("success") else []

async def get_lookup_tables():
    """Helper function to get lookup tables (dimension tables)."""
    result = await st.session_state.orchestrator.process({
        "workflow": "schema_configuration",
        "action": "get_lookup_tables"
    })
    return result.get("data", []) if result.get("success") else []

async def schema_configuration():
    st.header("Schema Configuration")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Table Categories", "Suggested Joins", "SQL Examples"])
    
    with tab1:
        with st.expander("Table Categories", expanded=False):
            try:
                # Get schema information through orchestrator
                schema_info = await st.session_state.orchestrator.process({
                    "workflow": "data_analysis",
                    "action": "get_schema_info"
                })
                
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
                schema_info = await st.session_state.orchestrator.process({
                    "workflow": "data_analysis",
                    "action": "get_schema_info"
                })
                
                if not schema_info.get("success"):
                    st.error("Failed to get schema information")
                    return
                
                tables = schema_info["data"]["tables"]
                all_join_patterns = []
                
                # Get join patterns for each table
                for table in tables:
                    join_patterns = await st.session_state.orchestrator.process({
                        "workflow": "schema_configuration",
                        "action": "get_join_patterns",
                        "parameters": {
                            "table_name": table["name"],
                            "schema_name": table["schema"]
                        }
                    })
                    
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