from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

class VisualizationAgent(BaseAgent):
    """Agent responsible for creating visualizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("visualization_agent", config)
        self.default_colors = px.colors.qualitative.Set3
        self.theme = config.get("theme", "plotly_white") if config else "plotly_white"
    
    async def initialize(self) -> bool:
        """Initialize visualization resources."""
        try:
            # Set default template
            import plotly.io as pio
            pio.templates.default = self.theme
            self.log_activity("initialized", {"theme": self.theme})
            return True
        except Exception as e:
            await self.handle_error(e, {"action": "initialize"})
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process visualization requests."""
        try:
            if not self.validate_input(input_data, ["type", "data"]):
                raise ValueError("Invalid input data")
            
            viz_type = input_data["type"]
            
            # Ensure data is converted to DataFrame if it's a list of dicts
            data = input_data["data"]
            if isinstance(data, list):
                data = pd.DataFrame(data)
            elif isinstance(data, dict) and "tables" in data:
                # Handle special case for table listings
                data = pd.DataFrame(data["tables"])
            else:
                data = pd.DataFrame(data)
            
            if viz_type == "dashboard":
                return await self._create_dashboard(data, input_data.get("config", {}))
            elif viz_type == "chart":
                return await self._create_chart(data, input_data.get("config", {}))
            elif viz_type == "animated":
                return await self._create_animated_chart(data, input_data.get("config", {}))
            else:
                raise ValueError(f"Unknown visualization type: {viz_type}")
                
        except Exception as e:
            self.logger.error(f"Error in visualization processing: {str(e)}")
            return await self.handle_error(e, {"input": input_data})
    
    async def cleanup(self) -> bool:
        """Cleanup visualization resources."""
        return True
    
    async def _create_dashboard(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a dashboard with multiple visualizations."""
        try:
            dashboard_type = config.get("dashboard_type", "sales")
            
            if dashboard_type == "sales":
                fig = await self._create_sales_dashboard(data, config)
            elif dashboard_type == "inventory":
                fig = await self._create_inventory_dashboard(data, config)
            elif dashboard_type == "customer":
                fig = await self._create_customer_dashboard(data, config)
            else:
                raise ValueError(f"Unknown dashboard type: {dashboard_type}")
            
            return {
                "success": True,
                "figure": fig,
                "type": "dashboard"
            }
            
        except Exception as e:
            return await self.handle_error(e, {"action": "create_dashboard"})
    
    async def _create_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a single chart."""
        try:
            chart_type = config.get("chart_type", "line")
            
            if chart_type == "line":
                fig = px.line(
                    data,
                    x=config.get("x"),
                    y=config.get("y"),
                    color=config.get("color"),
                    title=config.get("title", ""),
                    labels=config.get("labels", {})
                )
            elif chart_type == "bar":
                fig = px.bar(
                    data,
                    x=config.get("x"),
                    y=config.get("y"),
                    color=config.get("color"),
                    title=config.get("title", ""),
                    labels=config.get("labels", {})
                )
            elif chart_type == "scatter":
                fig = px.scatter(
                    data,
                    x=config.get("x"),
                    y=config.get("y"),
                    color=config.get("color"),
                    size=config.get("size"),
                    title=config.get("title", ""),
                    labels=config.get("labels", {})
                )
            else:
                raise ValueError(f"Unknown chart type: {chart_type}")
            
            # Apply common styling
            self._apply_styling(fig, config)
            
            return {
                "success": True,
                "figure": fig,
                "type": "chart"
            }
            
        except Exception as e:
            return await self.handle_error(e, {"action": "create_chart"})
    
    async def _create_animated_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an animated chart."""
        try:
            if not self.validate_input(config, ["animation_frame"]):
                raise ValueError("Animation frame not specified")
            
            fig = px.scatter(
                data,
                x=config.get("x"),
                y=config.get("y"),
                size=config.get("size"),
                color=config.get("color"),
                animation_frame=config["animation_frame"],
                title=config.get("title", ""),
                labels=config.get("labels", {})
            )
            
            # Customize animation
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = config.get("frame_duration", 1000)
            fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = config.get("transition_duration", 500)
            
            # Apply common styling
            self._apply_styling(fig, config)
            
            return {
                "success": True,
                "figure": fig,
                "type": "animated"
            }
            
        except Exception as e:
            return await self.handle_error(e, {"action": "create_animated_chart"})
    
    async def _create_sales_dashboard(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a sales dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Sales Trend",
                "Top Products",
                "Sales by Category",
                "Regional Distribution"
            )
        )
        
        # Sales Trend
        trend_data = data.groupby(config.get("time_column", "date")).sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=trend_data[config.get("time_column", "date")],
                y=trend_data[config.get("value_column", "sales")],
                name="Sales Trend"
            ),
            row=1, col=1
        )
        
        # Top Products
        top_products = data.groupby(config.get("product_column", "product"))\
            .sum()\
            .sort_values(config.get("value_column", "sales"), ascending=False)\
            .head(10)
        fig.add_trace(
            go.Bar(
                x=top_products.index,
                y=top_products[config.get("value_column", "sales")],
                name="Top Products"
            ),
            row=1, col=2
        )
        
        # Sales by Category
        category_sales = data.groupby(config.get("category_column", "category")).sum()
        fig.add_trace(
            go.Pie(
                labels=category_sales.index,
                values=category_sales[config.get("value_column", "sales")],
                name="Category Distribution"
            ),
            row=2, col=1
        )
        
        # Regional Distribution
        if config.get("region_column") in data.columns:
            region_sales = data.groupby(config["region_column"]).sum()
            fig.add_trace(
                go.Bar(
                    x=region_sales.index,
                    y=region_sales[config.get("value_column", "sales")],
                    name="Regional Sales"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Sales Dashboard"
        )
        
        return fig
    
    def _apply_styling(self, fig: go.Figure, config: Dict[str, Any]):
        """Apply common styling to a figure."""
        fig.update_layout(
            template=self.theme,
            title_x=0.5,
            title_font_size=20,
            showlegend=config.get("show_legend", True),
            height=config.get("height", 500),
            width=config.get("width", None),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        if config.get("x_title"):
            fig.update_xaxes(title_text=config["x_title"])
        if config.get("y_title"):
            fig.update_yaxes(title_text=config["y_title"]) 