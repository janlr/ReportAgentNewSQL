from typing import Dict, Any, Optional, List
import openai
import anthropic
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import hashlib
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from .base_agent import BaseAgent

# Optional prometheus import
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

class LLMManagerAgent(BaseAgent):
    """Agent responsible for managing LLM interactions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        super().__init__("llm_manager_agent", config)
        self.required_config = ["provider", "model", "api_key"]
        
        # Set core attributes first
        self.active_provider = config.get("provider", "anthropic")
        self.cache_dir = Path(config.get("cache_dir", "./cache/llm"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self._setup_clients()
        self._setup_cache()
        
        # Initialize optional components
        if self.config.get("enable_monitoring", False) and PROMETHEUS_AVAILABLE:
            self._setup_monitoring()
        
        # Initialize cost tracking
        self.daily_costs = {}
        self.cost_limits = config.get("cost_limits", {
            "daily": float("inf"),
            "monthly": float("inf")
        })
    
    def _setup_clients(self):
        """Initialize API clients."""
        provider = self.config.get("provider", "anthropic")
        if provider == "anthropic":
            api_key = self.config.get("api_key")
            self.logger.info("Setting up Anthropic client...")
            
            if not api_key:
                self.logger.error("No API key provided in config")
                return
                
            # Remove any quotes that might have been included
            api_key = api_key.strip('"').strip("'")
            
            if not api_key.startswith("sk-ant"):
                self.logger.error(f"API key format appears invalid - should start with 'sk-ant'")
                return
                
            try:
                self.client = anthropic.Client(api_key=api_key)
                self.logger.info("Anthropic client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Anthropic client: {str(e)}")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _generate_provider_response(self, provider: str, model: str, prompt: str, max_tokens: int) -> tuple[str, int]:
        """Generate response from specific provider."""
        if provider == "anthropic":
            response = await self.client.generate_content(prompt, max_tokens=max_tokens)
            # Approximate token count for Anthropic
            return response.text, len(prompt.split()) + len(response.text.split())
            
        raise ValueError(f"Unsupported provider: {provider}")

    async def _handle_cache_hit(self, cache_file: Path, provider: str, model: str) -> Optional[str]:
        """Handle cache hit and return cached response if valid."""
        try:
            with open(cache_file, "r") as f:
                cached_response = json.load(f)
            
            cache_time = datetime.fromisoformat(cached_response["timestamp"])
            if datetime.now() - cache_time < timedelta(seconds=self.cache_config["ttl"]):
                self._update_metrics("requests", {"provider": provider, "model": model, "cache_hit": "true"})
                return cached_response["response"]
        except Exception as e:
            self.logger.warning(f"Error reading cache: {str(e)}")
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_async(self, input_data: Dict[str, Any]) -> str:
        """Generate response from LLM with caching and monitoring."""
        try:
            provider = input_data.get("provider", self.active_provider)
            model = input_data.get("model", self.config["model"])
            prompt = input_data["prompt"]
            max_tokens = input_data.get("max_tokens", self.config["models"][model].get("max_tokens", 500))
            
            # Check cache
            if self.cache_config["enabled"]:
                cache_key = self._generate_cache_key(prompt, provider, model)
                cache_file = self.cache_dir / f"{cache_key}.json"
                
                if cache_file.exists():
                    if cached_response := await self._handle_cache_hit(cache_file, provider, model):
                        return cached_response
            
            # Generate response
            start_time = datetime.now()
            content, tokens = await self._generate_provider_response(provider, model, self._optimize_prompt(prompt), max_tokens)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Update metrics and costs
            self._update_metrics("requests", {"provider": provider, "model": model, "cache_hit": "false"})
            self._update_metrics("tokens", {"provider": provider, "model": model, "type": "total"}, tokens)
            self._update_metrics("duration", {"provider": provider, "model": model}, duration)
            await self._track_cost(provider, model, tokens)
            
            # Cache response
            if self.cache_config["enabled"]:
                await self._cache_response(cache_file, content, tokens, duration)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    async def _cache_response(self, cache_file: Path, content: str, tokens: int, duration: float):
        """Cache the response to file."""
        try:
            response_data = {
                "response": content,
                "timestamp": datetime.now().isoformat(),
                "tokens": tokens,
                "duration": duration
            }
            with open(cache_file, "w") as f:
                json.dump(response_data, f)
            
            self._update_metrics("cache_size", {"type": "responses"})
        except Exception as e:
            self.logger.warning(f"Error caching response: {str(e)}")

    def _setup_cache(self):
        """Initialize caching."""
        self.cache_config = self.config.get("cache", {
            "enabled": True,
            "ttl": 3600,  # 1 hour
            "max_size": 1000,
            "exact_match": True  # Use exact matching instead of semantic similarity
        })
        
        # Clean old cache files
        if self.cache_config["enabled"]:
            self._cleanup_old_cache()
    
    def _cleanup_old_cache(self):
        """Remove expired cache files."""
        try:
            current_time = datetime.now()
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                    cache_time = datetime.fromisoformat(cached_data["timestamp"])
                    if current_time - cache_time > timedelta(seconds=self.cache_config["ttl"]):
                        cache_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Error cleaning cache file {cache_file}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {str(e)}")
    
    def _setup_monitoring(self):
        """Initialize monitoring metrics if prometheus is available."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available. Monitoring disabled.")
            self.metrics = {}
            return

        self.metrics = {
            "requests": Counter(
                "llm_requests_total",
                "Total number of LLM requests",
                ["provider", "model", "cache_hit"]
            ),
            "tokens": Counter(
                "llm_tokens_total",
                "Total number of tokens processed",
                ["provider", "model", "type"]
            ),
            "duration": Histogram(
                "llm_request_duration_seconds",
                "Duration of LLM requests",
                ["provider", "model"]
            ),
            "cost": Counter(
                "llm_cost_total",
                "Total cost of LLM requests",
                ["provider", "model"]
            ),
            "cache_size": Gauge(
                "llm_cache_size",
                "Number of items in cache",
                ["type"]
            ),
            "daily_cost": Gauge(
                "llm_daily_cost",
                "Daily cost of LLM usage",
                ["provider"]
            )
        }
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        providers = []
        if "anthropic" in self.config:
            providers.append("anthropic")
        return providers
    
    def get_available_models(self, provider: str) -> List[str]:
        """Get available models for a provider."""
        if provider not in self.config:
            return []
        return list(self.config[provider].get("models", {}).keys())
    
    def set_active_provider(self, provider: str):
        """Set the active LLM provider."""
        if provider not in self.get_available_providers():
            raise ValueError(f"Provider {provider} not available")
        self.active_provider = provider
    
    def _generate_cache_key(self, prompt: str, provider: str, model: str) -> str:
        """Generate a cache key for a request."""
        key_components = [
            prompt,
            provider,
            model,
            str(self.config[provider]["models"][model].get("max_tokens", 500))
        ]
        return hashlib.md5("_".join(key_components).encode()).hexdigest()
    
    def _optimize_prompt(self, prompt: str, max_length: int = 1000) -> str:
        """Optimize prompt to reduce token usage."""
        # Simple length-based optimization
        if len(prompt) > max_length:
            lines = prompt.split("\n")
            # Keep first and last few lines
            kept_lines = lines[:3] + ["..."] + lines[-3:]
            return "\n".join(kept_lines)
        return prompt
    
    async def _track_cost(self, provider: str, model: str, tokens: int):
        """Track cost and check against limits."""
        model_config = self.config["models"].get(model, {})
        cost_per_1k = model_config.get("cost_per_1k_tokens", 0)
        cost = (tokens / 1000) * cost_per_1k
        
        # Update daily costs
        today = datetime.now().date().isoformat()
        self.daily_costs[today] = self.daily_costs.get(today, 0) + cost
        
        # Update metrics
        self._update_metrics("cost", {"provider": provider, "model": model}, cost)
        self._update_metrics("daily_cost", {"provider": provider}, self.daily_costs[today])
        
        # Check limits
        if self.daily_costs[today] > self.cost_limits["daily"]:
            raise Exception("Daily cost limit exceeded")
    
    async def initialize(self) -> bool:
        """Initialize the LLM manager agent."""
        try:
            # Check that all required configuration is present
            if not self.validate_config(self.required_config):
                self.logger.error("Missing required configuration fields for LLM manager")
                return False
            
            # Verify API keys are valid
            if self.active_provider == "anthropic":
                api_key = self.config.get("api_key", "")
                
                # Log key format for debugging (safely)
                key_start = api_key[:10] if len(api_key) > 10 else "N/A"
                key_length = len(api_key)
                self.logger.info(f"API Key format check - Start: {key_start}..., Length: {key_length}")
                
                # Check for common issues
                if not api_key:
                    self.logger.error("API key is empty")
                    return False
                    
                if '"' in api_key or "'" in api_key:
                    self.logger.warning("API key contains quotes - removing them")
                    api_key = api_key.strip('"').strip("'")
                    self.config["api_key"] = api_key
                
                if not api_key.startswith("sk-ant"):
                    self.logger.error("API key format is invalid - should start with 'sk-ant'")
                    return False
                
                try:
                    # Test connection to active provider
                    test_prompt = "Test connection."
                    self.logger.info("Testing Anthropic API connection...")
                    await self.generate_async({
                        "prompt": test_prompt,
                        "max_tokens": 5
                    })
                    self.logger.info("Successfully connected to Anthropic API")
                except Exception as e:
                    self.logger.error(f"Failed to connect to Anthropic API: {str(e)}")
                    self.logger.error(f"Exception type: {type(e).__name__}")
                    if "401" in str(e):
                        self.logger.error("Authentication failed - please check your API key")
                    elif "403" in str(e):
                        self.logger.error("Access forbidden - your API key may not be activated yet")
                    elif "429" in str(e):
                        self.logger.error("Rate limit exceeded - please try again later")
                    return False
            
            self.logger.info(f"LLM manager initialized successfully with provider: {self.active_provider}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM manager: {str(e)}")
            return False
            
    async def cleanup(self) -> bool:
        """Clean up resources."""
        try:
            # Clear cache if needed
            if self.cache_config["enabled"]:
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
            
            # Reset metrics
            for metric in self.metrics.values():
                if hasattr(metric, "clear"):
                    metric.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up LLM manager: {str(e)}")
            return False

    def _update_metrics(self, metric_name: str, labels: Dict[str, str], value: float = 1):
        """Update metrics if monitoring is enabled."""
        if hasattr(self, 'metrics') and metric_name in self.metrics:
            if metric_name == "duration":
                self.metrics[metric_name].labels(**labels).observe(value)
            else:
                self.metrics[metric_name].labels(**labels).inc(value)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return response."""
        try:
            if "prompt" not in input_data:
                raise ValueError("Prompt is required in input data")
                
            response = await self.generate_async(input_data)
            return {
                "success": True,
                "response": response,
                "provider": self.active_provider,
                "model": self.config.get("model")
            }
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 