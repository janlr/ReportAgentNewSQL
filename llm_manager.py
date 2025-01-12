from typing import Dict, Any, Optional, List
import openai
import anthropic
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from prometheus_client import Counter, Histogram, Gauge
import os
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMManager:
    """Manages LLM providers, caching, and optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_provider = config.get("default_provider", "openai")
        self.cache_dir = Path(config.get("cache_dir", "./cache/llm"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_clients()
        self._setup_cache()
        self._setup_monitoring()
        self._setup_logging()
        
        # Load embeddings model for semantic caching
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_cache = {}
        
        # Initialize cost tracking
        self.daily_costs = {}
        self.cost_limits = config.get("cost_limits", {
            "daily": float("inf"),
            "monthly": float("inf")
        })
    
    def _setup_clients(self):
        """Initialize API clients."""
        # OpenAI setup
        if "openai" in self.config:
            openai.api_key = self.config["openai"]["api_key"]
        
        # Anthropic setup
        if "anthropic" in self.config:
            self.anthropic_client = anthropic.Client(api_key=self.config["anthropic"]["api_key"])
        
        # Azure OpenAI setup
        if "azure_openai" in self.config:
            openai.api_type = "azure"
            openai.api_base = self.config["azure_openai"]["api_base"]
            openai.api_version = self.config["azure_openai"]["api_version"]
            openai.api_key = self.config["azure_openai"]["api_key"]
    
    def _setup_cache(self):
        """Initialize caching."""
        self.cache_config = self.config.get("cache", {
            "enabled": True,
            "ttl": 3600,  # 1 hour
            "max_size": 1000,
            "similarity_threshold": 0.95
        })
    
    def _setup_monitoring(self):
        """Initialize monitoring metrics."""
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
    
    def _setup_logging(self):
        """Initialize logging."""
        self.logger = logging.getLogger("llm_manager")
        self.logger.setLevel(logging.INFO)
    
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
        self.metrics["cost"].labels(provider=provider, model=model).inc(cost)
        self.metrics["daily_cost"].labels(provider=provider).set(self.daily_costs[today])
        
        # Check limits
        if self.daily_costs[today] > self.cost_limits["daily"]:
            raise Exception("Daily cost limit exceeded")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response from LLM with caching and monitoring."""
        try:
            provider = input_data.get("provider", self.active_provider)
            model = input_data.get("model", self.config[provider]["default_model"])
            prompt = input_data["prompt"]
            max_tokens = input_data.get("max_tokens", self.config[provider]["models"][model].get("max_tokens", 500))
            
            # Check cache if enabled
            if self.cache_config["enabled"]:
                cache_key = self._generate_cache_key(prompt, provider, model)
                cache_file = self.cache_dir / f"{cache_key}.json"
                
                if cache_file.exists():
                    with open(cache_file, "r") as f:
                        cached_response = json.load(f)
                    
                    # Check TTL
                    cache_time = datetime.fromisoformat(cached_response["timestamp"])
                    if datetime.now() - cache_time < timedelta(seconds=self.cache_config["ttl"]):
                        self.metrics["requests"].labels(
                            provider=provider,
                            model=model,
                            cache_hit="true"
                        ).inc()
                        return cached_response["response"]
            
            # Optimize prompt
            prompt = self._optimize_prompt(prompt)
            
            # Generate response
            start_time = datetime.now()
            
            if provider == "openai":
                response = await openai.ChatCompletion.acreate(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens
                
            elif provider == "anthropic":
                response = await self.anthropic_client.completion.create(
                    model=model,
                    prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=max_tokens
                )
                content = response.completion
                tokens = len(prompt.split()) + len(content.split())  # Approximate
                
            elif provider == "azure_openai":
                response = await openai.ChatCompletion.acreate(
                    engine=self.config["azure_openai"]["models"][model]["deployment_name"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.metrics["requests"].labels(
                provider=provider,
                model=model,
                cache_hit="false"
            ).inc()
            self.metrics["tokens"].labels(
                provider=provider,
                model=model,
                type="total"
            ).inc(tokens)
            self.metrics["duration"].labels(
                provider=provider,
                model=model
            ).observe(duration)
            
            # Track cost
            await self._track_cost(provider, model, tokens)
            
            result = {
                "content": content,
                "tokens": tokens,
                "duration": duration,
                "model": model,
                "provider": provider
            }
            
            # Cache result
            if self.cache_config["enabled"]:
                cache_data = {
                    "response": result,
                    "timestamp": datetime.now().isoformat()
                }
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f)
                
                self.metrics["cache_size"].labels(type="response").set(
                    len(list(self.cache_dir.glob("*.json")))
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous version of generate_async."""
        import asyncio
        return asyncio.run(self.generate_async(input_data))
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "daily_costs": self.daily_costs,
            "cache_size": len(list(self.cache_dir.glob("*.json"))),
            "embeddings_cache_size": len(self.embeddings_cache)
        }
    
    def clear_cache(self):
        """Clear both response and embedding caches."""
        # Clear response cache
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        
        # Clear embeddings cache
        self.embeddings_cache.clear()
        
        # Update metrics
        self.metrics["cache_size"].labels(type="response").set(0)
        self.metrics["cache_size"].labels(type="embeddings").set(0) 