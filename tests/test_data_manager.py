import pytest
import pandas as pd
import tempfile
from pathlib import Path
import os
from agents.data_manager import DataManager

@pytest.fixture
def data_manager():
    """Create a temporary DataManager instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield DataManager(storage_dir=temp_dir)

@pytest.fixture
def sample_excel_file():
    """Create a temporary Excel file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
        # Create sample DataFrame
        df1 = pd.DataFrame({
            "A": range(5),
            "B": range(5, 10)
        })
        df2 = pd.DataFrame({
            "X": range(3),
            "Y": range(3, 6)
        })
        
        # Write to Excel file
        with pd.ExcelWriter(temp_file.name) as writer:
            df1.to_excel(writer, sheet_name="Sheet1", index=False)
            df2.to_excel(writer, sheet_name="Sheet2", index=False)
        
        yield temp_file.name
        os.unlink(temp_file.name)

def test_import_excel_single_sheet(data_manager, sample_excel_file):
    """Test importing a single sheet from Excel."""
    result = data_manager.import_excel(
        sample_excel_file,
        sheet_name="Sheet1"
    )
    
    assert isinstance(result, dict)
    assert "Sheet1" in result
    assert isinstance(result["Sheet1"], pd.DataFrame)
    assert list(result["Sheet1"].columns) == ["A", "B"]
    assert len(result["Sheet1"]) == 5

def test_import_excel_multiple_sheets(data_manager, sample_excel_file):
    """Test importing multiple sheets from Excel."""
    result = data_manager.import_excel(
        sample_excel_file,
        sheet_name=["Sheet1", "Sheet2"]
    )
    
    assert isinstance(result, dict)
    assert "Sheet1" in result and "Sheet2" in result
    assert isinstance(result["Sheet1"], pd.DataFrame)
    assert isinstance(result["Sheet2"], pd.DataFrame)
    assert list(result["Sheet2"].columns) == ["X", "Y"]
    assert len(result["Sheet2"]) == 3

def test_add_favorite(data_manager):
    """Test adding a favorite report."""
    favorite_id = data_manager.add_favorite(
        name="Test Report",
        report_type="test",
        parameters={"param1": "value1"},
        description="Test description",
        tags=["tag1", "tag2"]
    )
    
    assert isinstance(favorite_id, str)
    
    # Verify the favorite was added
    favorite = data_manager.get_favorite(favorite_id)
    assert favorite["name"] == "Test Report"
    assert favorite["report_type"] == "test"
    assert favorite["parameters"] == {"param1": "value1"}
    assert favorite["description"] == "Test description"
    assert set(favorite["tags"]) == {"tag1", "tag2"}

def test_list_favorites(data_manager):
    """Test listing favorites with filters."""
    # Add some test favorites
    data_manager.add_favorite(
        name="Test Report 1",
        report_type="sales",
        parameters={"period": "monthly"},
        tags=["sales", "monthly"]
    )
    
    data_manager.add_favorite(
        name="Test Report 2",
        report_type="inventory",
        parameters={"period": "weekly"},
        tags=["inventory", "weekly"]
    )
    
    # Test filtering by report type
    sales_reports = data_manager.list_favorites(report_type="sales")
    assert len(sales_reports) == 1
    assert sales_reports[0]["name"] == "Test Report 1"
    
    # Test filtering by tags
    monthly_reports = data_manager.list_favorites(tags=["monthly"])
    assert len(monthly_reports) == 1
    assert "monthly" in monthly_reports[0]["tags"]

def test_delete_favorite(data_manager):
    """Test deleting a favorite report."""
    # Add a favorite
    favorite_id = data_manager.add_favorite(
        name="Test Report",
        report_type="test",
        parameters={"param1": "value1"}
    )
    
    # Verify it exists
    assert data_manager.get_favorite(favorite_id) is not None
    
    # Delete it
    data_manager.delete_favorite(favorite_id)
    
    # Verify it was deleted
    with pytest.raises(Exception):
        data_manager.get_favorite(favorite_id)

def test_favorite_usage_tracking(data_manager):
    """Test that favorite usage is tracked correctly."""
    # Add a favorite
    favorite_id = data_manager.add_favorite(
        name="Test Report",
        report_type="test",
        parameters={"param1": "value1"}
    )
    
    # Access it multiple times
    for _ in range(3):
        favorite = data_manager.get_favorite(favorite_id)
    
    # Verify use count
    favorite = data_manager.get_favorite(favorite_id)
    assert favorite["use_count"] == 4  # Initial get + 3 accesses

def test_invalid_excel_file(data_manager):
    """Test handling of invalid Excel files."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
        temp_file.write(b"Not an Excel file")
        temp_file.flush()
        
        with pytest.raises(Exception):
            data_manager.import_excel(temp_file.name)

def test_duplicate_favorite_tags(data_manager):
    """Test handling of duplicate tags when adding favorites."""
    favorite_id = data_manager.add_favorite(
        name="Test Report",
        report_type="test",
        parameters={"param1": "value1"},
        tags=["tag1", "tag1", "tag2"]  # Duplicate tag
    )
    
    favorite = data_manager.get_favorite(favorite_id)
    assert len(favorite["tags"]) == 2  # Duplicates should be removed
    assert set(favorite["tags"]) == {"tag1", "tag2"} 