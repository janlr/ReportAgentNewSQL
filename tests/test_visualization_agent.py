import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import plotly.graph_objects as go
import plotly.express as px

from agents.visualization_agent import VisualizationAgent

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'sales': np.random.randint(100, 1000, 10),
        'category': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        'region': ['North', 'South', 'North', 'South', 'East', 'West', 'North', 'South', 'East', 'West']
    })

@pytest.fixture
def sample_config():
    return {
        "output_dir": "./test_visualizations",
        "theme": {
            "colorway": ["#1f77b4", "#ff7f0e", "#2ca02c"],
            "template": "plotly_white",
            "font": {"family": "Arial"}
        }
    }

@pytest.fixture
def agent(sample_config):
    return VisualizationAgent(sample_config)

class TestVisualizationCreation:
    """Test cases for creating different types of visualizations."""
    
    def test_create_line_chart(self, agent, sample_data):
        """Test creation of line charts."""
        fig = agent.create_line_chart(
            data=sample_data,
            x='date',
            y='sales',
            color='category',
            title='Sales Over Time'
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Sales Over Time'
        assert len(fig.data) == len(sample_data['category'].unique())
    
    def test_create_bar_chart(self, agent, sample_data):
        """Test creation of bar charts."""
        fig = agent.create_bar_chart(
            data=sample_data,
            x='category',
            y='sales',
            color='region',
            title='Sales by Category'
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Sales by Category'
        assert fig.layout.barmode == 'group'
    
    def test_create_scatter_plot(self, agent, sample_data):
        """Test creation of scatter plots."""
        fig = agent.create_scatter_plot(
            data=sample_data,
            x='sales',
            y='sales',
            color='category',
            size='sales',
            title='Sales Correlation'
        )
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Sales Correlation'
        assert 'marker.size' in str(fig.data[0])

class TestDashboardCreation:
    """Test cases for creating dashboards with multiple visualizations."""
    
    def test_create_dashboard(self, agent, sample_data):
        """Test creation of a dashboard with multiple plots."""
        plots = [
            {
                'type': 'line',
                'data': sample_data,
                'x': 'date',
                'y': 'sales',
                'title': 'Sales Trend'
            },
            {
                'type': 'bar',
                'data': sample_data,
                'x': 'category',
                'y': 'sales',
                'title': 'Sales by Category'
            }
        ]
        
        dashboard = agent.create_dashboard(
            plots=plots,
            layout='vertical',
            title='Sales Dashboard'
        )
        
        assert isinstance(dashboard, go.Figure)
        assert len(dashboard.data) > 1
        assert dashboard.layout.title.text == 'Sales Dashboard'
    
    def test_dashboard_layout_options(self, agent, sample_data):
        """Test different dashboard layout options."""
        plots = [
            {'type': 'line', 'data': sample_data, 'x': 'date', 'y': 'sales'},
            {'type': 'bar', 'data': sample_data, 'x': 'category', 'y': 'sales'}
        ]
        
        # Test horizontal layout
        dashboard_h = agent.create_dashboard(plots=plots, layout='horizontal')
        assert dashboard_h.layout.grid.rows == 1
        assert dashboard_h.layout.grid.cols == 2
        
        # Test vertical layout
        dashboard_v = agent.create_dashboard(plots=plots, layout='vertical')
        assert dashboard_v.layout.grid.rows == 2
        assert dashboard_v.layout.grid.cols == 1

class TestAnimatedVisualizations:
    """Test cases for creating animated visualizations."""
    
    def test_create_animated_scatter(self, agent, sample_data):
        """Test creation of animated scatter plots."""
        fig = agent.create_animated_scatter(
            data=sample_data,
            x='sales',
            y='sales',
            color='category',
            animation_frame='date',
            title='Sales Animation'
        )
        
        assert isinstance(fig, go.Figure)
        assert 'frames' in fig
        assert len(fig.frames) > 0
    
    def test_create_animated_bar(self, agent, sample_data):
        """Test creation of animated bar charts."""
        fig = agent.create_animated_bar(
            data=sample_data,
            x='category',
            y='sales',
            color='region',
            animation_frame='date',
            title='Sales by Category Animation'
        )
        
        assert isinstance(fig, go.Figure)
        assert 'frames' in fig
        assert len(fig.frames) > 0

class TestThemeAndStyling:
    """Test cases for visualization theming and styling."""
    
    def test_apply_theme(self, agent, sample_data):
        """Test application of custom theme to visualizations."""
        fig = agent.create_line_chart(
            data=sample_data,
            x='date',
            y='sales'
        )
        
        assert fig.layout.template == agent.config['theme']['template']
        assert fig.layout.font.family == agent.config['theme']['font']['family']
    
    def test_custom_color_palette(self, agent, sample_data):
        """Test application of custom color palette."""
        fig = agent.create_bar_chart(
            data=sample_data,
            x='category',
            y='sales',
            color='region'
        )
        
        assert fig.layout.colorway == agent.config['theme']['colorway']

class TestExportAndSaving:
    """Test cases for exporting and saving visualizations."""
    
    def test_save_figure(self, agent, sample_data, tmp_path):
        """Test saving figures in different formats."""
        fig = agent.create_line_chart(
            data=sample_data,
            x='date',
            y='sales'
        )
        
        # Test HTML export
        html_path = tmp_path / "test_plot.html"
        success = agent.save_figure(fig, html_path, format='html')
        assert success is True
        assert html_path.exists()
        
        # Test PNG export
        png_path = tmp_path / "test_plot.png"
        success = agent.save_figure(fig, png_path, format='png')
        assert success is True
        assert png_path.exists()
    
    def test_save_dashboard(self, agent, sample_data, tmp_path):
        """Test saving dashboards."""
        plots = [
            {'type': 'line', 'data': sample_data, 'x': 'date', 'y': 'sales'},
            {'type': 'bar', 'data': sample_data, 'x': 'category', 'y': 'sales'}
        ]
        dashboard = agent.create_dashboard(plots=plots)
        
        path = tmp_path / "dashboard.html"
        success = agent.save_dashboard(dashboard, path)
        assert success is True
        assert path.exists()

if __name__ == "__main__":
    pytest.main([__file__]) 