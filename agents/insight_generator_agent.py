from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from pathlib import Path

class InsightGeneratorAgent(BaseAgent):
    """Agent responsible for generating insights and summaries using LLM."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("insight_generator", config)
        self.llm_manager = config.get("llm_manager")
        self.cache_dir = Path(config.get("cache_dir", "./cache/insights"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize insight generation resources."""
        try:
            if not self.llm_manager:
                raise ValueError("LLM manager not provided")
            
            self.log_activity("initialized", {"llm_provider": self.llm_manager.active_provider})
            return True
        except Exception as e:
            await self.handle_error(e, {"action": "initialize"})
            return False
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process insight generation requests."""
        try:
            if not self.validate_input(input_data, ["action"]):
                raise ValueError("Invalid input data")
            
            action = input_data["action"]
            if action == "generate_summary":
                return await self._generate_summary(input_data)
            elif action == "generate_insights":
                return await self._generate_insights(input_data)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            return await self.handle_error(e, {"input": input_data})
    
    async def cleanup(self) -> bool:
        """Cleanup insight generation resources."""
        return True
    
    async def _generate_summary(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the data and results."""
        if not self.validate_input(input_data, ["data", "metadata"]):
            raise ValueError("Data or metadata not provided")
        
        try:
            data = pd.DataFrame(input_data["data"])
            metadata = input_data["metadata"]
            report_type = input_data.get("report_type", "general")
            
            # Generate cache key
            cache_key = self._generate_cache_key(data, metadata, report_type)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Check cache
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    cached_result = json.load(f)
                return {
                    "success": True,
                    "summary": cached_result["summary"],
                    "from_cache": True
                }
            
            # Prepare data statistics
            stats = self._calculate_statistics(data)
            
            # Generate prompt based on report type
            prompt = self._generate_prompt(report_type, stats, metadata)
            
            # Generate summary using LLM
            llm_response = await self.llm_manager.generate_async({
                "prompt": prompt,
                "max_tokens": 500
            })
            
            summary = {
                "overview": llm_response["content"],
                "key_findings": self._extract_key_findings(llm_response["content"]),
                "recommendations": self._extract_recommendations(llm_response["content"]),
                "generated_at": datetime.now().isoformat()
            }
            
            # Cache the result
            with open(cache_file, "w") as f:
                json.dump({"summary": summary}, f)
            
            return {
                "success": True,
                "summary": summary,
                "from_cache": False
            }
            
        except Exception as e:
            return await self.handle_error(e, {"action": "generate_summary"})
    
    async def _generate_insights(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed insights from the data."""
        if not self.validate_input(input_data, ["data"]):
            raise ValueError("Data not provided")
        
        try:
            data = pd.DataFrame(input_data["data"])
            insight_type = input_data.get("insight_type", "comprehensive")
            
            insights = []
            
            if insight_type in ["basic", "comprehensive"]:
                # Statistical insights
                stats_insights = self._generate_statistical_insights(data)
                insights.extend(stats_insights)
            
            if insight_type == "comprehensive":
                # Trend insights
                if self._has_time_column(data):
                    trend_insights = await self._generate_trend_insights(data)
                    insights.extend(trend_insights)
                
                # Correlation insights
                if len(data.select_dtypes(include=[np.number]).columns) >= 2:
                    correlation_insights = self._generate_correlation_insights(data)
                    insights.extend(correlation_insights)
                
                # Anomaly insights
                anomaly_insights = self._generate_anomaly_insights(data)
                insights.extend(anomaly_insights)
            
            return {
                "success": True,
                "insights": insights
            }
            
        except Exception as e:
            return await self.handle_error(e, {"action": "generate_insights"})
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics from the data."""
        stats = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "numeric_columns": {},
            "categorical_columns": {}
        }
        
        for col in data.select_dtypes(include=[np.number]).columns:
            stats["numeric_columns"][col] = {
                "mean": data[col].mean(),
                "median": data[col].median(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max()
            }
        
        for col in data.select_dtypes(exclude=[np.number]).columns:
            stats["categorical_columns"][col] = {
                "unique_values": data[col].nunique(),
                "top_values": data[col].value_counts().head(5).to_dict()
            }
        
        return stats
    
    def _generate_prompt(self, report_type: str, stats: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Generate a prompt for the LLM based on report type and statistics."""
        prompts = {
            "sales": """
                Analyze the following sales data statistics and provide:
                1. A clear overview of the sales performance
                2. Key findings and trends
                3. Actionable recommendations for improvement
                
                Statistics:
                {stats}
                
                Additional Context:
                {metadata}
                
                Format your response with clear sections for Overview, Key Findings, and Recommendations.
                Focus on business impact and actionable insights.
            """,
            "inventory": """
                Analyze the following inventory data statistics and provide:
                1. An overview of the inventory status
                2. Key findings about stock levels and movement
                3. Recommendations for optimization
                
                Statistics:
                {stats}
                
                Additional Context:
                {metadata}
                
                Format your response with clear sections for Overview, Key Findings, and Recommendations.
                Focus on efficiency and cost-saving opportunities.
            """,
            "general": """
                Analyze the following data statistics and provide:
                1. A clear overview of the main patterns and trends
                2. Key findings from the data
                3. Actionable recommendations
                
                Statistics:
                {stats}
                
                Additional Context:
                {metadata}
                
                Format your response with clear sections for Overview, Key Findings, and Recommendations.
                Focus on practical insights and actionable steps.
            """
        }
        
        prompt_template = prompts.get(report_type, prompts["general"])
        return prompt_template.format(
            stats=json.dumps(stats, indent=2),
            metadata=json.dumps(metadata, indent=2)
        )
    
    def _extract_key_findings(self, content: str) -> List[str]:
        """Extract key findings from the LLM response."""
        try:
            # Find the section between "Key Findings" and "Recommendations"
            start = content.find("Key Findings")
            end = content.find("Recommendations")
            
            if start == -1 or end == -1:
                return []
            
            findings_text = content[start:end]
            # Split by bullet points or numbers and clean up
            findings = [
                f.strip("- ").strip()
                for f in findings_text.split("\n")
                if f.strip().startswith(("-", "•", "*")) or (f.strip()[0].isdigit() and f.strip()[1] == ".")
            ]
            
            return findings
            
        except Exception:
            return []
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from the LLM response."""
        try:
            # Find the "Recommendations" section
            start = content.find("Recommendations")
            if start == -1:
                return []
            
            recommendations_text = content[start:]
            # Split by bullet points or numbers and clean up
            recommendations = [
                r.strip("- ").strip()
                for r in recommendations_text.split("\n")
                if r.strip().startswith(("-", "•", "*")) or (r.strip()[0].isdigit() and r.strip()[1] == ".")
            ]
            
            return recommendations
            
        except Exception:
            return []
    
    def _generate_cache_key(self, data: pd.DataFrame, metadata: Dict[str, Any], report_type: str) -> str:
        """Generate a cache key based on data characteristics and metadata."""
        key_components = [
            str(len(data)),
            str(sorted(data.columns.tolist())),
            str(sorted(metadata.items())),
            report_type
        ]
        
        return hashlib.md5("_".join(key_components).encode()).hexdigest()
    
    def _has_time_column(self, data: pd.DataFrame) -> bool:
        """Check if the DataFrame has a time-based column."""
        return any(
            str(dtype).startswith(("datetime", "timestamp"))
            for dtype in data.dtypes
        ) 