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
        super().__init__(config)
        self.required_config = ["provider", "model"]
        
        # Set up OpenAI API key if using OpenAI
        if config["provider"].lower() == "openai":
            openai.api_key = config.get("api_key")
        
        self.active_provider = config.get("default_provider", "openai")
        self.cache_dir = Path(config.get("cache_dir", "./cache/llm"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger first for better error tracking
        self._setup_logging()
        
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
        providers_config = {
            "openai": self._setup_openai,
            "anthropic": self._setup_anthropic,
            "azure_openai": self._setup_azure_openai
        }
        
        for provider, setup_func in providers_config.items():
            if provider in self.config:
                setup_func()
    
    def _setup_openai(self):
        """Setup OpenAI client."""
        if api_key := self.config["openai"].get("api_key"):
            openai.api_key = api_key
        else:
            self.logger.warning("OpenAI API key not provided")
    
    def _setup_anthropic(self):
        """Setup Anthropic client."""
        if api_key := self.config["anthropic"].get("api_key"):
            self.anthropic_client = anthropic.Client(api_key=api_key)
        else:
            self.logger.warning("Anthropic API key not provided")
    
    def _setup_azure_openai(self):
        """Setup Azure OpenAI client."""
        config = self.config["azure_openai"]
        if all(key in config for key in ["api_key", "api_base", "api_version"]):
            openai.api_type = "azure"
            openai.api_base = config["api_base"]
            openai.api_version = config["api_version"]
            openai.api_key = config["api_key"]
        else:
            self.logger.warning("Azure OpenAI configuration incomplete")

    async def _generate_provider_response(self, provider: str, model: str, prompt: str, max_tokens: int) -> tuple[str, int]:
        """Generate response from specific provider."""
        if provider == "openai":
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content, response.usage.total_tokens
            
        elif provider == "anthropic":
            response = await self.anthropic_client.completion.create(
                model=model,
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                max_tokens_to_sample=max_tokens
            )
            # Approximate token count for Anthropic
            return response.completion, len(prompt.split()) + len(response.completion.split())
            
        elif provider == "azure_openai":
            response = await openai.ChatCompletion.acreate(
                engine=self.config["azure_openai"]["models"][model]["deployment_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content, response.usage.total_tokens
        
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
            model = input_data.get("model", self.config[provider]["default_model"])
            prompt = input_data["prompt"]
            max_tokens = input_data.get("max_tokens", self.config[provider]["models"][model].get("max_tokens", 500))
            
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
        if "openai" in self.config:
            providers.append("openai")
        if "anthropic" in self.config:
            providers.append("anthropic")
        if "azure_openai" in self.config:
            providers.append("azure_openai")
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
        if provider not in self.config:
            return
        
        model_config = self.config[provider]["models"].get(model, {})
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
        """Initialize LLM manager."""
        if not self.validate_config(self.required_config):
            return False
        return True
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process LLM-related requests."""
        try:
            action = request.get("action")
            if not action:
                return {"success": False, "error": "No action specified"}
            
            if action == "generate_text":
                return await self._generate_text(request.get("parameters", {}))
            elif action == "analyze_text":
                return await self._analyze_text(request.get("parameters", {}))
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            self.logger.error(f"Error processing LLM request: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_text(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text using LLM."""
        # TODO: Implement actual text generation
        return {
            "success": True,
            "data": {
                "text": "Generated text placeholder"
            }
        }
    
    async def _analyze_text(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text using LLM."""
        # TODO: Implement actual text analysis
        return {
            "success": True,
            "data": {
                "analysis": "Text analysis placeholder"
            }
        }

    def _update_metrics(self, metric_name: str, labels: Dict[str, str], value: float = 1):
        """Update metrics if monitoring is enabled."""
        if hasattr(self, 'metrics') and metric_name in self.metrics:
            if metric_name == "duration":
                self.metrics[metric_name].labels(**labels).observe(value)
            else:
                self.metrics[metric_name].labels(**labels).inc(value) 