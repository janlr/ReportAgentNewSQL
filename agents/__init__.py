"""Agent-based reporting system powered by AutoGen."""

__version__ = "0.1.0"

# Import base agent first
from .base_agent import BaseAgent

# Define exports
__all__ = [
    'BaseAgent',
    'DatabaseAgent',
    'DataManagerAgent',
    'UserInterfaceAgent',
    'ReportGeneratorAgent',
    'InsightGeneratorAgent',
    'AssistantAgent',
    'MasterOrchestratorAgent'
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in __all__:
        if name == 'DatabaseAgent':
            from .database_agent import DatabaseAgent
            return DatabaseAgent
        elif name == 'DataManagerAgent':
            from .data_manager_agent import DataManagerAgent
            return DataManagerAgent
        elif name == 'UserInterfaceAgent':
            from .user_interface_agent import UserInterfaceAgent
            return UserInterfaceAgent
        elif name == 'ReportGeneratorAgent':
            from .report_generator_agent import ReportGeneratorAgent
            return ReportGeneratorAgent
        elif name == 'InsightGeneratorAgent':
            from .insight_generator_agent import InsightGeneratorAgent
            return InsightGeneratorAgent
        elif name == 'AssistantAgent':
            from .assistant_agent import AssistantAgent
            return AssistantAgent
        elif name == 'MasterOrchestratorAgent':
            from .master_orchestrator_agent import MasterOrchestratorAgent
            return MasterOrchestratorAgent
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 