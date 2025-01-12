import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch
import streamlit as st

from agents.user_interface_agent import UserInterfaceAgent

@pytest.fixture
def sample_config():
    return {
        "preferences_dir": "./test_preferences",
        "default_preferences": {
            "theme": "light",
            "date_format": "YYYY-MM-DD",
            "chart_type": "line",
            "page_size": 10
        }
    }

@pytest.fixture
def agent(sample_config):
    return UserInterfaceAgent(sample_config)

class TestUIInitialization:
    """Test cases for UI initialization and preferences management."""
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.preferences_dir == Path("./test_preferences")
        assert agent.default_preferences == agent.config["default_preferences"]
    
    @patch('pathlib.Path.exists')
    @patch('json.load')
    def test_load_preferences(self, mock_json_load, mock_exists, agent):
        """Test loading of user preferences."""
        mock_exists.return_value = True
        mock_preferences = {
            "theme": "dark",
            "date_format": "DD/MM/YYYY"
        }
        mock_json_load.return_value = mock_preferences
        
        preferences = agent.load_preferences()
        assert preferences["theme"] == "dark"
        assert preferences["date_format"] == "DD/MM/YYYY"
    
    @patch('pathlib.Path.exists')
    def test_load_default_preferences(self, mock_exists, agent):
        """Test loading of default preferences when no file exists."""
        mock_exists.return_value = False
        
        preferences = agent.load_preferences()
        assert preferences == agent.default_preferences

class TestPageRendering:
    """Test cases for page rendering functionality."""
    
    @patch('streamlit.title')
    def test_render_page_title(self, mock_title, agent):
        """Test rendering of page title."""
        agent.render_page({
            "type": "title",
            "content": "Test Page"
        })
        mock_title.assert_called_once_with("Test Page")
    
    @patch('streamlit.text')
    def test_render_text(self, mock_text, agent):
        """Test rendering of text content."""
        agent.render_page({
            "type": "text",
            "content": "Test content"
        })
        mock_text.assert_called_once_with("Test content")
    
    @patch('streamlit.dataframe')
    def test_render_dataframe(self, mock_dataframe, agent):
        """Test rendering of dataframe."""
        data = {"col1": [1, 2], "col2": ["a", "b"]}
        agent.render_page({
            "type": "dataframe",
            "content": data
        })
        mock_dataframe.assert_called_once()

class TestUserInput:
    """Test cases for handling user input."""
    
    @patch('streamlit.text_input')
    def test_text_input(self, mock_text_input, agent):
        """Test handling of text input."""
        mock_text_input.return_value = "test input"
        
        result = agent.handle_input({
            "type": "text_input",
            "label": "Enter text",
            "key": "test_input"
        })
        
        assert result == "test input"
        mock_text_input.assert_called_once_with(
            label="Enter text",
            key="test_input"
        )
    
    @patch('streamlit.selectbox')
    def test_select_input(self, mock_selectbox, agent):
        """Test handling of select input."""
        options = ["Option 1", "Option 2"]
        mock_selectbox.return_value = "Option 1"
        
        result = agent.handle_input({
            "type": "select",
            "label": "Choose option",
            "options": options,
            "key": "test_select"
        })
        
        assert result == "Option 1"
        mock_selectbox.assert_called_once()

class TestSessionManagement:
    """Test cases for session state management."""
    
    def test_set_session_data(self, agent):
        """Test setting session state data."""
        agent.set_session_data("test_key", "test_value")
        assert st.session_state.get("test_key") == "test_value"
    
    def test_get_session_data(self, agent):
        """Test retrieving session state data."""
        st.session_state["test_key"] = "test_value"
        value = agent.get_session_data("test_key")
        assert value == "test_value"
    
    def test_clear_session_data(self, agent):
        """Test clearing session state data."""
        st.session_state["test_key"] = "test_value"
        agent.clear_session_data("test_key")
        assert "test_key" not in st.session_state

class TestPreferencesManagement:
    """Test cases for user preferences management."""
    
    @patch('json.dump')
    @patch('pathlib.Path.exists')
    def test_save_preferences(self, mock_exists, mock_json_dump, agent):
        """Test saving user preferences."""
        mock_exists.return_value = True
        preferences = {
            "theme": "dark",
            "date_format": "DD/MM/YYYY"
        }
        
        success = agent.save_preferences(preferences)
        assert success is True
        mock_json_dump.assert_called_once()
    
    def test_update_preferences(self, agent):
        """Test updating user preferences."""
        current_preferences = agent.default_preferences.copy()
        updates = {
            "theme": "dark",
            "chart_type": "bar"
        }
        
        updated = agent.update_preferences(updates)
        assert updated["theme"] == "dark"
        assert updated["chart_type"] == "bar"
        assert updated["date_format"] == current_preferences["date_format"]

class TestErrorHandling:
    """Test cases for error handling in UI operations."""
    
    @patch('streamlit.error')
    def test_display_error(self, mock_error, agent):
        """Test error message display."""
        error_message = "Test error message"
        agent.display_error(error_message)
        mock_error.assert_called_once_with(error_message)
    
    @patch('pathlib.Path.exists')
    def test_preferences_file_error(self, mock_exists, agent):
        """Test handling of preferences file error."""
        mock_exists.return_value = False
        
        # Should fall back to default preferences
        preferences = agent.load_preferences()
        assert preferences == agent.default_preferences

class TestComponentRendering:
    """Test cases for rendering specific UI components."""
    
    @patch('streamlit.plotly_chart')
    def test_render_visualization(self, mock_plotly_chart, agent):
        """Test rendering of visualization components."""
        fig = Mock()
        agent.render_page({
            "type": "visualization",
            "content": fig
        })
        mock_plotly_chart.assert_called_once_with(fig, use_container_width=True)
    
    @patch('streamlit.download_button')
    def test_render_download_button(self, mock_download_button, agent):
        """Test rendering of download button."""
        agent.render_page({
            "type": "download_button",
            "label": "Download Report",
            "data": "test data",
            "file_name": "report.csv"
        })
        mock_download_button.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__]) 