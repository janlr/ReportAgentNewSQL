from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from .base_agent import BaseAgent
from .llm_manager_agent import LLMManagerAgent
from autogen import Agent

class InsightGeneratorAgent(BaseAgent):
    """Agent responsible for generating insights and summaries using LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        super().__init__("insight_generator_agent", config)
        self.required_config = ["llm_manager", "cache_dir"]
        self.cache_dir = Path(config.get("cache_dir", "./cache/insights"))
        self.llm_available = False
        
        # Validate LLM manager configuration
        if "llm_manager" not in config:
            raise ValueError("LLM manager configuration is required")
        
        self.llm_config = config["llm_manager"]
        if not all(k in self.llm_config for k in ["provider", "model"]):
            raise ValueError("LLM manager configuration must include provider and model")
        
        # Initialize LLM manager
        self.llm_manager = LLMManagerAgent({
            "provider": self.llm_config["provider"],
            "model": self.llm_config["model"],
            "api_key": self.llm_config["api_key"],
            "cache_dir": self.llm_config.get("cache_dir", "./cache/llm"),
            "cache": self.llm_config.get("cache", {
                "enabled": True,
                "ttl": 3600,
                "max_size": 1000,
                "exact_match": True
            }),
            "models": self.llm_config.get("models", {}),
            "monitoring": self.llm_config.get("monitoring", {"enabled": False}),
            "cost_limits": self.llm_config.get("cost_limits", {
                "daily": float("inf"),
                "monthly": float("inf")
            })
        })
            
    async def initialize(self) -> bool:
        """Initialize the insight generator agent."""
        if not self.validate_config(self.required_config):
            self.logger.error("Invalid configuration for insight generator agent")
            return False
            
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip LLM manager initialization - marked as unavailable
        self.llm_available = False
        self.logger.warning("Skipping LLM manager initialization - insights will be limited to statistical analysis only")
        
        self.logger.info("Insight generator agent initialized successfully (without LLM capabilities)")
        return True
    
    async def cleanup(self) -> bool:
        """Clean up insight generator resources."""
        try:
            # Clean up LLM manager if it was initialized
            if self.llm_available:
                try:
                    if not await self.llm_manager.cleanup():
                        self.logger.warning("Failed to clean up LLM manager")
                except Exception as e:
                    self.logger.warning(f"Error cleaning up LLM manager: {str(e)}")
            
            self.logger.info("Insight generator agent cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up insight generator agent: {str(e)}")
            return False
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process insight generation requests."""
        try:
            action = request.get("action")
            if not action:
                return {"success": False, "error": "No action specified"}
            
            # Check if LLM is required for this action
            llm_required_actions = ["generate_llm_insights", "summarize_with_llm"]
            if not self.llm_available and action in llm_required_actions:
                self.logger.warning(f"Cannot perform action {action} because LLM manager is not available")
                return {
                    "success": False, 
                    "error": "LLM manager not available. This feature is disabled.",
                    "fallback": True
                }
            
            if action == "generate_insights":
                return await self._generate_insights(request.get("parameters", {}))
            elif action == "analyze_trends":
                return await self._analyze_trends(request.get("parameters", {}))
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            self.logger.error(f"Error processing insight request: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_insights(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from data."""
        # TODO: Implement actual insight generation
        return {
            "success": True,
            "data": {
                "insights": [],
                "recommendations": []
            }
        }
    
    async def _analyze_trends(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in the data."""
        # TODO: Implement actual trend analysis
        return {
            "success": True,
            "data": {
                "trends": [],
                "patterns": []
            }
        }
    
    def _generate_summary(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the data with insights."""
        try:
            summary = {
                "summary": "",
                "findings": "",
                "recommendations": ""
            }
            
            # Basic statistics
            num_rows = len(df)
            num_cols = len(df.columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            date_cols = df.select_dtypes(include=['datetime64']).columns
            
            # Generate summary text
            summary["summary"] = f"""
            Dataset Overview:
            - Total records: {num_rows:,}
            - Total features: {num_cols}
            - Numeric features: {len(numeric_cols)}
            - Categorical features: {len(categorical_cols)}
            - Date features: {len(date_cols)}
            """
            
            # Generate findings
            findings = []
            
            # Analyze numeric columns
            for col in numeric_cols:
                stats_dict = df[col].describe()
                
                # Check for outliers
                Q1 = stats_dict['25%']
                Q3 = stats_dict['75%']
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
                
                if len(outliers) > 0:
                    findings.append(f"- {col} has {len(outliers)} outliers ({(len(outliers)/num_rows*100):.1f}% of data)")
                
                # Check for skewness
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    findings.append(f"- {col} shows {'positive' if skewness > 0 else 'negative'} skewness ({skewness:.2f})")
            
            # Analyze categorical columns
            for col in categorical_cols:
                unique_vals = df[col].nunique()
                if unique_vals == 1:
                    findings.append(f"- {col} has only one unique value")
                elif unique_vals/num_rows > 0.9:
                    findings.append(f"- {col} has high cardinality ({unique_vals} unique values)")
            
            # Analyze date columns
            for col in date_cols:
                date_range = df[col].max() - df[col].min()
                findings.append(f"- {col} spans {date_range.days} days")
            
            summary["findings"] = "\n".join(findings)
            
            # Generate recommendations
            recommendations = []
            
            # Recommendations based on findings
            if any("outliers" in finding for finding in findings):
                recommendations.append("- Consider handling outliers through removal or transformation")
            
            if any("skewness" in finding for finding in findings):
                recommendations.append("- Consider applying transformations to handle skewed distributions")
            
            if any("high cardinality" in finding for finding in findings):
                recommendations.append("- Consider feature engineering or dimensionality reduction for high cardinality features")
            
            summary["recommendations"] = "\n".join(recommendations)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return {
                "summary": "Error generating summary",
                "findings": str(e),
                "recommendations": "Please check the data and try again"
            }
    
    def _analyze_data(self, df: pd.DataFrame, analysis_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform specific analysis on the data."""
        try:
            if analysis_type == "correlation":
                # Correlation analysis
                numeric_df = df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                
                # Find strong correlations
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corr.append({
                                "feature1": corr_matrix.columns[i],
                                "feature2": corr_matrix.columns[j],
                                "correlation": corr_matrix.iloc[i, j]
                            })
                
                return {
                    "correlation_matrix": corr_matrix.to_dict(),
                    "strong_correlations": strong_corr
                }
            
            elif analysis_type == "time_series":
                date_col = parameters.get("date_column")
                value_col = parameters.get("value_column")
                
                if not date_col or not value_col:
                    raise ValueError("Date and value columns are required for time series analysis")
                
                # Resample data
                df[date_col] = pd.to_datetime(df[date_col])
                resampled = df.set_index(date_col)[value_col].resample('D').mean()
                
                # Calculate trends
                rolling_mean = resampled.rolling(window=7).mean()
                rolling_std = resampled.rolling(window=7).std()
                
                return {
                    "original_data": resampled.to_dict(),
                    "trend": rolling_mean.to_dict(),
                    "volatility": rolling_std.to_dict()
                }
            
            elif analysis_type == "distribution":
                column = parameters.get("column")
                
                if not column:
                    raise ValueError("Column name is required for distribution analysis")
                
                # Calculate distribution statistics
                stats_dict = df[column].describe()
                skewness = df[column].skew()
                kurtosis = df[column].kurtosis()
                
                # Perform normality test
                _, p_value = stats.normaltest(df[column].dropna())
                
                return {
                    "statistics": stats_dict.to_dict(),
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "is_normal": p_value > 0.05,
                    "p_value": p_value
                }
            
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing data: {str(e)}")
            return {}
    
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

class MasterOrchestratorAgent:
    def __init__(self):
        self.name = "Master Orchestrator"
        # other initialization code
    
    def coordinate_workflow(self):
        # workflow coordination code
        pass

    def handle_errors(self):
        # error handling code
        pass 