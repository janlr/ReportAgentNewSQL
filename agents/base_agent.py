from typing import Dict, Any, Optional, List
import logging
from abc import ABC, abstractmethod
from pathlib import Path

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize with name and configuration."""
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.required_config = []
    
    def validate_config(self, required_fields: list) -> bool:
        """Validate that all required configuration fields are present."""
        # Skip validation if no required fields
        if not required_fields:
            return True
            
        # For database config, handle Windows Authentication case
        if self.config.get("trusted_connection", "yes").lower() == "yes":
            # Remove user and password from required fields if using Windows Auth
            required_fields = [f for f in required_fields if f not in ["user", "password"]]
            
        missing_fields = [field for field in required_fields if field not in self.config]
        if missing_fields:
            self.logger.error(f"Missing required configuration fields: {missing_fields}")
            return False
        return True
    
    async def initialize(self) -> bool:
        """Initialize the agent."""
        return True
    
    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request and return the result."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources used by the agent."""
        pass
    
    def log_activity(self, activity: str, details: Dict[str, Any]):
        """Log agent activity."""
        self.logger.info(f"{activity}: {details}")
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate input data has required fields."""
        return all(field in input_data for field in required_fields)
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors during agent processing."""
        self.logger.error(f"Error in {self.__class__.__name__}: {str(error)}", extra=context)
        return {
            "success": False,
            "error": str(error),
            "context": context
        } 