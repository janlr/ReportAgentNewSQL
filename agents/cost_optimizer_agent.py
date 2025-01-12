from typing import Dict, List, Optional
import logging
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CostOptimizerAgent(BaseAgent):
    def __init__(self, config: Dict):
        """
        Initialize the Cost Optimizer Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.cache = {}
        self.usage_stats = {
            "api_calls": [],
            "query_executions": [],
            "report_generations": []
        }
        
    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            # Create cache directory if it doesn't exist
            cache_dir = Path(self.config.get("cache_dir", "./cache"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CostOptimizerAgent: {str(e)}")
            return False
            
    async def process(self, data: Dict) -> Dict:
        """
        Process incoming requests.
        
        Args:
            data: Dictionary containing the request parameters
            
        Returns:
            Dictionary containing the processing results
        """
        action = data.get("action")
        if action == "track_api":
            self.track_api_usage(
                data.get("api_name"),
                data.get("tokens_used"),
                data.get("duration_ms"),
                data.get("cost")
            )
            return {"result": "success"}
        elif action == "track_query":
            self.track_query_execution(
                data.get("query"),
                data.get("duration_ms"),
                data.get("rows_processed")
            )
            return {"result": "success"}
        elif action == "track_report":
            self.track_report_generation(
                data.get("report_type"),
                data.get("size_bytes"),
                data.get("duration_ms"),
                data.get("resources_used", {})
            )
            return {"result": "success"}
        elif action == "get_summary":
            return {
                "result": self.get_usage_summary(
                    data.get("start_time"),
                    data.get("end_time")
                )
            }
        elif action == "get_recommendations":
            return {"result": self.get_optimization_recommendations()}
        elif action == "optimize_cache":
            self.optimize_cache_usage(
                data.get("max_cache_size_mb", 100)
            )
            return {"result": "success"}
        else:
            raise ValueError(f"Unknown action: {action}")
            
    async def cleanup(self):
        """Cleanup resources."""
        # Save usage stats to file
        try:
            stats_file = Path(self.config.get("stats_file", "./usage_stats.json"))
            with open(stats_file, "w") as f:
                json.dump(self.usage_stats, f)
        except Exception as e:
            logger.error(f"Error saving usage stats: {str(e)}")
        
    def track_api_usage(
        self,
        api_name: str,
        tokens_used: int,
        duration_ms: int,
        cost: float
    ):
        """
        Track API usage statistics.
        
        Args:
            api_name: Name of the API (e.g., 'openai', 'azure')
            tokens_used: Number of tokens consumed
            duration_ms: Duration of the API call in milliseconds
            cost: Estimated cost of the API call
        """
        self.usage_stats["api_calls"].append({
            "timestamp": datetime.now().isoformat(),
            "api_name": api_name,
            "tokens_used": tokens_used,
            "duration_ms": duration_ms,
            "cost": cost
        })
        
    def track_query_execution(
        self,
        query: str,
        duration_ms: int,
        rows_processed: int
    ):
        """
        Track database query execution statistics.
        
        Args:
            query: SQL query executed
            duration_ms: Query execution time in milliseconds
            rows_processed: Number of rows processed
        """
        self.usage_stats["query_executions"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "duration_ms": duration_ms,
            "rows_processed": rows_processed
        })
        
    def track_report_generation(
        self,
        report_type: str,
        size_bytes: int,
        duration_ms: int,
        resources_used: Dict
    ):
        """
        Track report generation statistics.
        
        Args:
            report_type: Type of report generated
            size_bytes: Size of the generated report
            duration_ms: Generation time in milliseconds
            resources_used: Dictionary of resources consumed
        """
        self.usage_stats["report_generations"].append({
            "timestamp": datetime.now().isoformat(),
            "report_type": report_type,
            "size_bytes": size_bytes,
            "duration_ms": duration_ms,
            "resources_used": resources_used
        })
        
    def get_usage_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """
        Get usage statistics summary.
        
        Args:
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            Dictionary containing usage statistics
        """
        # Filter stats by time range if provided
        api_calls = self._filter_by_time_range(
            self.usage_stats["api_calls"],
            start_time,
            end_time
        )
        
        query_executions = self._filter_by_time_range(
            self.usage_stats["query_executions"],
            start_time,
            end_time
        )
        
        report_generations = self._filter_by_time_range(
            self.usage_stats["report_generations"],
            start_time,
            end_time
        )
        
        # Calculate summaries
        api_summary = {
            "total_calls": len(api_calls),
            "total_tokens": sum(call["tokens_used"] for call in api_calls),
            "total_cost": sum(call["cost"] for call in api_calls),
            "avg_duration_ms": sum(call["duration_ms"] for call in api_calls) / len(api_calls) if api_calls else 0
        }
        
        query_summary = {
            "total_queries": len(query_executions),
            "total_rows": sum(q["rows_processed"] for q in query_executions),
            "avg_duration_ms": sum(q["duration_ms"] for q in query_executions) / len(query_executions) if query_executions else 0
        }
        
        report_summary = {
            "total_reports": len(report_generations),
            "total_size_bytes": sum(r["size_bytes"] for r in report_generations),
            "avg_duration_ms": sum(r["duration_ms"] for r in report_generations) / len(report_generations) if report_generations else 0
        }
        
        return {
            "api_usage": api_summary,
            "query_usage": query_summary,
            "report_usage": report_summary,
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            }
        }
        
    def get_optimization_recommendations(self) -> List[Dict]:
        """
        Generate cost optimization recommendations.
        
        Returns:
            List of recommendations with explanations
        """
        recommendations = []
        
        # Analyze API usage patterns
        api_calls = self.usage_stats["api_calls"]
        if api_calls:
            # Check for high token usage
            avg_tokens = sum(call["tokens_used"] for call in api_calls) / len(api_calls)
            if avg_tokens > 1000:  # Arbitrary threshold
                recommendations.append({
                    "type": "api_optimization",
                    "priority": "high",
                    "description": "High average token usage detected",
                    "suggestion": "Consider implementing token optimization strategies",
                    "potential_savings": "10-20% reduction in API costs"
                })
        
        # Analyze query patterns
        query_executions = self.usage_stats["query_executions"]
        if query_executions:
            # Check for slow queries
            slow_queries = [
                q for q in query_executions
                if q["duration_ms"] > 1000  # Queries taking more than 1 second
            ]
            if slow_queries:
                recommendations.append({
                    "type": "query_optimization",
                    "priority": "medium",
                    "description": f"Found {len(slow_queries)} slow queries",
                    "suggestion": "Consider adding indexes or optimizing query structure",
                    "potential_savings": "20-30% reduction in query execution time"
                })
        
        # Analyze report generation patterns
        report_generations = self.usage_stats["report_generations"]
        if report_generations:
            # Check for large reports
            large_reports = [
                r for r in report_generations
                if r["size_bytes"] > 10 * 1024 * 1024  # Reports larger than 10MB
            ]
            if large_reports:
                recommendations.append({
                    "type": "report_optimization",
                    "priority": "low",
                    "description": f"Found {len(large_reports)} large reports",
                    "suggestion": "Consider implementing report size optimization or compression",
                    "potential_savings": "40-50% reduction in storage costs"
                })
        
        return recommendations
        
    def optimize_cache_usage(self, max_cache_size_mb: int = 100):
        """
        Optimize cache usage by removing old or less frequently used items.
        
        Args:
            max_cache_size_mb: Maximum cache size in megabytes
        """
        current_size = sum(len(json.dumps(v)) for v in self.cache.values())
        current_size_mb = current_size / (1024 * 1024)
        
        if current_size_mb > max_cache_size_mb:
            # Remove oldest items first
            items = sorted(
                self.cache.items(),
                key=lambda x: x[1].get("last_accessed", 0)
            )
            
            while current_size_mb > max_cache_size_mb and items:
                key, _ = items.pop(0)
                removed_size = len(json.dumps(self.cache[key]))
                del self.cache[key]
                current_size_mb -= removed_size / (1024 * 1024)
                
    def _filter_by_time_range(
        self,
        items: List[Dict],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Filter items by time range."""
        if not (start_time or end_time):
            return items
            
        filtered_items = []
        for item in items:
            timestamp = datetime.fromisoformat(item["timestamp"])
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            filtered_items.append(item)
            
        return filtered_items 