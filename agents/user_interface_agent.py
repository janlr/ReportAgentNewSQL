from typing import Dict, Any, List, Optional, Union
import streamlit as st
import json
import logging
from pathlib import Path
from datetime import datetime
from .base_agent import BaseAgent
import plotly.express as px
import plotly.graph_objects as go
import os

class UserInterfaceAgent(BaseAgent):
    """Agent responsible for user interface interactions and rendering."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        super().__init__("user_interface_agent")
        self.config = config
        self.required_config = ["preferences_dir"]
        
        # Set up preferences directory
        self.preferences_dir = Path(config.get("preferences_dir", "./preferences"))
        
        # Initialize session data
        if "session_data" not in st.session_state:
            st.session_state.session_data = {}
        
        # Load user preferences
        self.preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load user preferences from file."""
        if self.preferences_dir.exists():
            try:
                with open(self.preferences_dir / "user_preferences.json", "r") as f:
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
            with open(self.preferences_dir / "user_preferences.json", "w") as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving preferences: {str(e)}")
    
    async def initialize(self) -> bool:
        """Initialize user interface resources."""
        if not self.validate_config(self.required_config):
            return False
            
        try:
            # Create preferences directory
            self.preferences_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("User interface agent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing user interface agent: {str(e)}")
            return False
    
    async def cleanup(self) -> bool:
        """Clean up user interface resources."""
        try:
            # Clean up any resources
            self._save_preferences()
            self.logger.info("User interface agent cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up user interface agent: {str(e)}")
            return False
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process user interface requests."""
        try:
            if not self.validate_input(request, ["action"]):
                raise ValueError("Missing required field: action")
            
            action = request.get("action")
            
            if action == "get_preferences":
                return await self._get_preferences()
            elif action == "save_preferences":
                return await self._save_preferences(request.get("parameters", {}))
            elif action == "render_page":
                page = request.get("page")
                components = request.get("components", [])
                
                if not page:
                    raise ValueError("Page title is required for render_page action")
                
                rendered_page = self._render_page(page, components)
                return {
                    "success": True,
                    "data": rendered_page
                }
                
            elif action == "create_visualization":
                data = request.get("data")
                viz_type = request.get("viz_type")
                parameters = request.get("parameters", {})
                
                if not data or not viz_type:
                    raise ValueError("Data and visualization type are required")
                
                visualization = self._create_visualization(data, viz_type, parameters)
                return {
                    "success": True,
                    "data": visualization
                }
            
            elif action == "handle_input":
                input_type = request.get("type")
                key = request.get("key")
                value = request.get("value")
                return await self._handle_input(input_type, key, value)
            
            elif action == "update_preferences":
                preferences = request.get("preferences")
                return await self._update_preferences(preferences)
            
            elif action == "get_session_data":
                key = request.get("key")
                return await self._get_session_data(key)
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            self.logger.error(f"Error processing UI request: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _render_page(self, page_title: str, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Render a page with the given components."""
        try:
            st.title(page_title)
            
            rendered_components = []
            for component in components:
                comp_type = component.get("type")
                
                if comp_type == "header":
                    st.header(component.get("text", ""))
                    rendered_components.append({"type": "header", "id": id(component)})
                
                elif comp_type == "subheader":
                    st.subheader(component.get("text", ""))
                    rendered_components.append({"type": "subheader", "id": id(component)})
                
                elif comp_type == "text":
                    st.text(component.get("text", ""))
                    rendered_components.append({"type": "text", "id": id(component)})
                
                elif comp_type == "markdown":
                    st.markdown(component.get("text", ""))
                    rendered_components.append({"type": "markdown", "id": id(component)})
                
                elif comp_type == "dataframe":
                    st.dataframe(component.get("data"))
                    rendered_components.append({"type": "dataframe", "id": id(component)})
                
                elif comp_type == "chart":
                    st.plotly_chart(component.get("figure"))
                    rendered_components.append({"type": "chart", "id": id(component)})
                
                elif comp_type == "metrics":
                    cols = st.columns(len(component.get("metrics", [])))
                    for col, metric in zip(cols, component.get("metrics", [])):
                        with col:
                            st.metric(
                                label=metric.get("label", ""),
                                value=metric.get("value", ""),
                                delta=metric.get("delta")
                            )
                    rendered_components.append({"type": "metrics", "id": id(component)})
            
            return {
                "page_title": page_title,
                "components": rendered_components,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering page: {str(e)}")
            return {}
    
    def _create_visualization(self, data: Dict[str, Any], viz_type: str, parameters: Dict[str, Any]) -> go.Figure:
        """Create a visualization based on the specified type and parameters."""
        try:
            if viz_type == "line":
                fig = px.line(
                    data,
                    x=parameters.get("x"),
                    y=parameters.get("y"),
                    title=parameters.get("title", ""),
                    labels=parameters.get("labels", {}),
                    color=parameters.get("color"),
                    line_dash=parameters.get("line_dash"),
                    markers=parameters.get("markers", True)
                )
            
            elif viz_type == "bar":
                fig = px.bar(
                    data,
                    x=parameters.get("x"),
                    y=parameters.get("y"),
                    title=parameters.get("title", ""),
                    labels=parameters.get("labels", {}),
                    color=parameters.get("color"),
                    barmode=parameters.get("barmode", "group")
                )
            
            elif viz_type == "scatter":
                fig = px.scatter(
                    data,
                    x=parameters.get("x"),
                    y=parameters.get("y"),
                    title=parameters.get("title", ""),
                    labels=parameters.get("labels", {}),
                    color=parameters.get("color"),
                    size=parameters.get("size"),
                    hover_data=parameters.get("hover_data", [])
                )
            
            elif viz_type == "pie":
                fig = px.pie(
                    data,
                    values=parameters.get("values"),
                    names=parameters.get("names"),
                    title=parameters.get("title", ""),
                    hole=parameters.get("hole", 0)
                )
            
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            # Update layout based on parameters
            layout_updates = parameters.get("layout", {})
            fig.update_layout(**layout_updates)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return go.Figure()
    
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
    
    async def _get_preferences(self) -> Dict[str, Any]:
        """Get user preferences."""
        # TODO: Implement actual preferences retrieval
        return {
            "success": True,
            "data": {}
        }
    
    async def _save_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Save user preferences."""
        # TODO: Implement actual preferences saving
        return {
            "success": True,
            "data": preferences
        } 