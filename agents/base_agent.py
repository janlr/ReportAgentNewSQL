from typing import Dict, Any, Optional, List
import logging
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent with required resources."""
        pass
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results."""
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
        self.logger.error(f"Error in {self.name}: {str(error)}", extra=context)
        return {
            "success": False,
            "error": str(error),
            "context": context
        } 