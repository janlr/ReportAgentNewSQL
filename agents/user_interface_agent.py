from typing import Dict, Any, List, Optional, Union
import streamlit as st
import json
import logging
from pathlib import Path
from datetime import datetime
from .base_agent import BaseAgent

class UserInterfaceAgent(BaseAgent):
    """Agent responsible for user interface interactions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("user_interface_agent")
        self.config = config
        
        # Set up preferences directory
        self.preferences_dir = Path(config.get("preferences_dir", "./preferences"))
        self.preferences_dir.mkdir(parents=True, exist_ok=True)
        self.preferences_file = self.preferences_dir / "user_preferences.json"
        
        # Initialize session data
        if "session_data" not in st.session_state:
            st.session_state.session_data = {}
        
        # Load user preferences
        self.preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load user preferences from file."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading preferences: {str(e)}")
        
        return {
            "theme": "light",
            "default_report_type": "sales",
            "chart_preferences": {
                "color_scheme": "default",
                "show_grid": True,
                "interactive": True
            },
            "export_preferences": {
                "default_format": "pdf",
                "include_summary": True,
                "include_insights": True
            },
            "notification_preferences": {
                "email_reports": False,
                "alert_on_completion": True
            }
        }
    
    def _save_preferences(self):
        """Save user preferences to file."""
        try:
            with open(self.preferences_file, "w") as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving preferences: {str(e)}")
    
    async def initialize(self) -> bool:
        """Initialize the user interface."""
        try:
            self.logger.info("Initializing user interface")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing user interface: {str(e)}")
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user interface requests."""
        action = input_data.get("action")
        
        if not action:
            raise ValueError("Action not specified in input data")
        
        try:
            if action == "render_page":
                page = input_data.get("page")
                components = input_data.get("components", [])
                return await self._render_page(page, components)
            
            elif action == "handle_input":
                input_type = input_data.get("type")
                key = input_data.get("key")
                value = input_data.get("value")
                return await self._handle_input(input_type, key, value)
            
            elif action == "update_preferences":
                preferences = input_data.get("preferences")
                return await self._update_preferences(preferences)
            
            elif action == "get_session_data":
                key = input_data.get("key")
                return await self._get_session_data(key)
            
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing UI request: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up user interface resources."""
        self._save_preferences()
        self.logger.info("Cleaned up user interface resources")
    
    async def _render_page(self, page: str, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Render a page with specified components."""
        try:
            # Set page configuration
            st.set_page_config(
                page_title=f"Report Agent - {page}",
                page_icon="📊",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Render components
            rendered_components = []
            for comp in components:
                comp_type = comp.get("type")
                
                if comp_type == "header":
                    st.header(comp.get("text", ""))
                    rendered_components.append({"type": "header", "status": "success"})
                
                elif comp_type == "text":
                    st.text(comp.get("text", ""))
                    rendered_components.append({"type": "text", "status": "success"})
                
                elif comp_type == "markdown":
                    st.markdown(comp.get("text", ""))
                    rendered_components.append({"type": "markdown", "status": "success"})
                
                elif comp_type == "input":
                    value = st.text_input(
                        label=comp.get("label", ""),
                        value=comp.get("default", ""),
                        key=comp.get("key")
                    )
                    rendered_components.append({
                        "type": "input",
                        "key": comp.get("key"),
                        "value": value,
                        "status": "success"
                    })
                
                elif comp_type == "number":
                    value = st.number_input(
                        label=comp.get("label", ""),
                        min_value=comp.get("min"),
                        max_value=comp.get("max"),
                        value=comp.get("default"),
                        key=comp.get("key")
                    )
                    rendered_components.append({
                        "type": "number",
                        "key": comp.get("key"),
                        "value": value,
                        "status": "success"
                    })
                
                elif comp_type == "date":
                    value = st.date_input(
                        label=comp.get("label", ""),
                        value=comp.get("default"),
                        key=comp.get("key")
                    )
                    rendered_components.append({
                        "type": "date",
                        "key": comp.get("key"),
                        "value": value.isoformat() if value else None,
                        "status": "success"
                    })
                
                elif comp_type == "select":
                    value = st.selectbox(
                        label=comp.get("label", ""),
                        options=comp.get("options", []),
                        index=comp.get("default_index", 0),
                        key=comp.get("key")
                    )
                    rendered_components.append({
                        "type": "select",
                        "key": comp.get("key"),
                        "value": value,
                        "status": "success"
                    })
                
                elif comp_type == "multiselect":
                    value = st.multiselect(
                        label=comp.get("label", ""),
                        options=comp.get("options", []),
                        default=comp.get("default", []),
                        key=comp.get("key")
                    )
                    rendered_components.append({
                        "type": "multiselect",
                        "key": comp.get("key"),
                        "value": value,
                        "status": "success"
                    })
                
                elif comp_type == "checkbox":
                    value = st.checkbox(
                        label=comp.get("label", ""),
                        value=comp.get("default", False),
                        key=comp.get("key"),
                        help=comp.get("help")
                    )
                    rendered_components.append({
                        "type": "checkbox",
                        "key": comp.get("key"),
                        "value": value,
                        "status": "success"
                    })
                
                elif comp_type == "button":
                    if st.button(
                        label=comp.get("label", ""),
                        key=comp.get("key")
                    ):
                        rendered_components.append({
                            "type": "button",
                            "key": comp.get("key"),
                            "clicked": True,
                            "status": "success"
                        })
                    else:
                        rendered_components.append({
                            "type": "button",
                            "key": comp.get("key"),
                            "clicked": False,
                            "status": "success"
                        })
                
                elif comp_type == "plotly":
                    st.plotly_chart(
                        comp.get("figure"),
                        use_container_width=comp.get("use_container_width", True)
                    )
                    rendered_components.append({"type": "plotly", "status": "success"})
                
                elif comp_type == "dataframe":
                    st.dataframe(
                        comp.get("data"),
                        use_container_width=comp.get("use_container_width", True)
                    )
                    rendered_components.append({"type": "dataframe", "status": "success"})
                
                elif comp_type == "error":
                    st.error(comp.get("text", ""))
                    rendered_components.append({"type": "error", "status": "success"})
                
                elif comp_type == "success":
                    st.success(comp.get("text", ""))
                    rendered_components.append({"type": "success", "status": "success"})
                
                elif comp_type == "info":
                    st.info(comp.get("text", ""))
                    rendered_components.append({"type": "info", "status": "success"})
                
                elif comp_type == "warning":
                    st.warning(comp.get("text", ""))
                    rendered_components.append({"type": "warning", "status": "success"})
                
                else:
                    self.logger.warning(f"Unknown component type: {comp_type}")
                    rendered_components.append({
                        "type": comp_type,
                        "status": "error",
                        "message": "Unknown component type"
                    })
            
            # Log page view
            self.logger.info(f"Rendered page: {page}")
            
            return {
                "page": page,
                "components": rendered_components,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering page: {str(e)}")
            raise
    
    async def _handle_input(self, input_type: str, key: str, value: Any) -> Dict[str, Any]:
        """Handle user input and store in session data."""
        try:
            if not key:
                raise ValueError("Input key not specified")
            
            # Store value in session data
            st.session_state.session_data[key] = value
            
            self.logger.info(f"Handled input: {key} = {value}")
            return {
                "type": input_type,
                "key": key,
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error handling input: {str(e)}")
            raise
    
    async def _update_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences."""
        try:
            if not preferences:
                raise ValueError("Preferences not specified")
            
            # Update preferences
            self.preferences.update(preferences)
            self._save_preferences()
            
            self.logger.info("Updated user preferences")
            return {
                "preferences": self.preferences,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating preferences: {str(e)}")
            raise
    
    async def _get_session_data(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get session data for a specific key or all data."""
        try:
            if key:
                value = st.session_state.session_data.get(key)
                if value is None:
                    raise ValueError(f"Session data not found for key: {key}")
                
                return {
                    "key": key,
                    "value": value,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "data": st.session_state.session_data,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting session data: {str(e)}")
            raise 