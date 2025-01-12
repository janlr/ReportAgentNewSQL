import streamlit as st
import json
import logging
from pathlib import Path
from datetime import datetime
from agents import MasterOrchestratorAgent
from llm_manager import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    try:
        with open("config.py", "r") as f:
            config = {}
            exec(f.read(), {}, config)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

# Initialize components
async def init_components():
    try:
        config = load_config()
        
        # Initialize LLM manager
        llm_manager = LLMManager(config.get("llm", {}))
        
        # Initialize master orchestrator
        orchestrator = MasterOrchestratorAgent(config)
        success = await orchestrator.initialize()
        
        if not success:
            raise RuntimeError("Failed to initialize master orchestrator")
        
        return orchestrator, llm_manager
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

# Main application
def main():
    st.set_page_config(
        page_title="Report Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator, st.session_state.llm_manager = asyncio.run(init_components())
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Analysis",
        ["Sales Analysis", "Inventory Analysis", "Customer Analysis", "Database Config"]
    )
    
    # Database configuration check
    if page != "Database Config":
        db_status = asyncio.run(st.session_state.orchestrator.process({
            "workflow": "database_management",
            "action": "test_connection"
        }))
        
        if not db_status["result"].get("success"):
            st.error("Please configure database connection first")
            page = "Database Config"
    
    try:
        if page == "Sales Analysis":
            # Get date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
            
            # Get categories
            categories = st.multiselect(
                "Product Categories",
                ["Category A", "Category B", "Category C"]
            )
            
            # Generate summary option
            generate_summary = st.checkbox(
                "Generate Summary",
                help="Use AI to generate a summary of the analysis"
            )
            
            if st.button("Generate Report"):
                result = asyncio.run(st.session_state.orchestrator.process({
                    "workflow": "report_generation",
                    "report_type": "sales",
                    "parameters": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "categories": categories
                    },
                    "generate_summary": generate_summary
                }))
                
                # Save to favorites option
                if st.checkbox("Save to Favorites"):
                    name = st.text_input("Report Name")
                    description = st.text_area("Description")
                    tags = st.multiselect("Tags", ["monthly", "quarterly", "annual", "custom"])
                    
                    if st.button("Save"):
                        asyncio.run(st.session_state.orchestrator.process({
                            "workflow": "data_management",
                            "action": "add_favorite",
                            "name": name,
                            "description": description,
                            "report_type": "sales",
                            "parameters": {
                                "start_date": start_date.isoformat(),
                                "end_date": end_date.isoformat(),
                                "categories": categories
                            },
                            "tags": tags
                        }))
                        st.success("Analysis saved to favorites!")
        
        elif page == "Inventory Analysis":
            # Similar structure for inventory analysis
            pass
            
        elif page == "Customer Analysis":
            # Similar structure for customer analysis
            pass
            
        elif page == "Database Config":
            st.header("Database Configuration")
            
            # Database type selection
            db_type = st.selectbox(
                "Database Type",
                ["SQL Server", "MySQL", "PostgreSQL", "SQLite"]
            )
            
            # Connection details
            server = st.text_input("Server")
            database = st.text_input("Database")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Test Connection"):
                result = asyncio.run(st.session_state.orchestrator.process({
                    "workflow": "database_management",
                    "action": "test_connection",
                    "parameters": {
                        "type": db_type,
                        "server": server,
                        "database": database,
                        "username": username,
                        "password": password
                    }
                }))
                
                if result["result"].get("success"):
                    st.success("Connection successful!")
                else:
                    st.error(f"Connection failed: {result['result'].get('error')}")
    
    except Exception as e:
        logger.error(f"Error in application: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 