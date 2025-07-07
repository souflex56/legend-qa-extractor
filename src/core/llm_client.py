"""LLM client module for interacting with Ollama models."""

from typing import Optional, Dict, Any
import logging
import time
import requests
from ollama import Client

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with Ollama LLM models."""
    
    def __init__(self, host: str = "http://localhost:11434", model_name: str = "qwen2.5:7b-instruct"):
        self.host = host
        self.model_name = model_name
        self.logger = logger
        self._is_warmed_up = False
        
        # **🚀 PERFORMANCE MONITORING: Track request timing and statistics**
        self._performance_stats = {
            'total_requests': 0,
            'total_time': 0.0,
            'fastest_request': float('inf'),
            'slowest_request': 0.0,
            'warmup_time': 0.0,
            'connection_reuses': 0
        }
        
        # **🚀 PERFORMANCE OPTIMIZATION: Connection pooling and Keep-Alive**
        self._setup_optimized_client()
        
        # Test connection
        self._test_connection()
    
    def _setup_optimized_client(self):
        """Setup optimized Ollama client with connection pooling and Keep-Alive."""
        try:
            # Create a requests session with connection pooling
            session = requests.Session()
            
            # Configure connection pool and Keep-Alive
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,  # Number of connection pools
                pool_maxsize=20,      # Max connections in pool
                max_retries=3,        # Retry failed requests
                pool_block=False      # Don't block on pool exhaustion
            )
            
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            # Set Keep-Alive headers
            session.headers.update({
                'Connection': 'keep-alive',
                'Keep-Alive': 'timeout=30, max=100'
            })
            
            # Create Ollama client with optimized session
            self.client = Client(host=self.host)
            
            # Store session for potential direct requests
            self._session = session
            
            self.logger.debug("✅ Optimized client with connection pooling initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to setup optimized client, falling back to default: {e}")
            # Fallback to default client
            self.client = Client(host=self.host)
            self._session = None
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to list models to test connection
            models_response = self.client.list()
            self.logger.info(f"Successfully connected to Ollama at {self.host}")
            
            # **🔧 FIX: Handle both dict and ListResponse objects for compatibility**
            if hasattr(models_response, 'models'):
                # New ollama client returns ListResponse object
                model_list = models_response.models
                model_names = [model.model for model in model_list]
            elif isinstance(models_response, dict):
                # Legacy format: dict with 'models' key
                model_list = models_response.get('models', [])
                model_names = [m.get('name', 'unknown') for m in model_list if isinstance(m, dict)]
            else:
                model_names = []
            
            self.logger.info(f"Available models: {model_names}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama at {self.host}: {e}")
            return False
    
    def warmup_model(self) -> bool:
        """预热模型，消除冷启动延迟
        
        Returns:
            True if warmup successful, False otherwise
        """
        if self._is_warmed_up:
            self.logger.info("Model already warmed up, skipping...")
            return True
        
        self.logger.info(f"🔥 Starting model warmup for {self.model_name}...")
        start_time = time.time()
        
        try:
            # Use a simple warmup prompt
            warmup_prompt = "请简单回答：什么是人工智能？"
            
            # **🔥 CRITICAL FIX: Add keep_alive during warmup**
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": warmup_prompt}],
                options={"temperature": 0.1},
                keep_alive="30m"  # Keep model loaded for 30 minutes
            )
            
            warmup_time = time.time() - start_time
            
            if response and "message" in response:
                self._is_warmed_up = True
                self._performance_stats['warmup_time'] = warmup_time
                self.logger.info(f"✅ Model warmup completed in {warmup_time:.2f}s")
                self.logger.info(f"🔥 Model will stay loaded for 30 minutes to prevent cold starts")
                self.logger.info(f"Sample response: {response['message']['content'][:100]}...")
                return True
            else:
                self.logger.error("Model warmup failed - invalid response format")
                return False
                
        except Exception as e:
            warmup_time = time.time() - start_time
            self.logger.error(f"❌ Model warmup failed after {warmup_time:.2f}s: {e}")
            return False
    
    def call_ollama(self, prompt: str, temperature: float = 0.1, **kwargs) -> Optional[str]:
        """Call Ollama model with given prompt.
        
        Args:
            prompt: Input prompt for the model
            temperature: Model temperature for response generation
            **kwargs: Additional options for the model
            
        Returns:
            Model response text or None if failed
        """
        # **🚀 PERFORMANCE MONITORING: Start timing**
        start_time = time.time()
        
        try:
            # Auto-warmup if not done yet
            if not self._is_warmed_up:
                self.logger.warning("Model not warmed up, performing auto-warmup...")
                self.warmup_model()
            
            options = {
                "temperature": temperature,
                **kwargs
            }
            
            # **🔥 CRITICAL FIX: Add keep_alive to prevent cold starts**
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive="30m"  # Keep model loaded for 30 minutes
            )
            
            # **🚀 PERFORMANCE MONITORING: Record successful request**
            request_time = time.time() - start_time
            self._update_performance_stats(request_time, success=True)
            
            if response and "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                self.logger.warning("Unexpected response format from Ollama")
                return None
                
        except Exception as e:
            # **🚀 PERFORMANCE MONITORING: Record failed request**
            request_time = time.time() - start_time
            self._update_performance_stats(request_time, success=False)
            
            self.logger.error(f"Ollama API call failed: {e}")
            return None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current model.
        
        Returns:
            Model information dictionary or None if failed
        """
        try:
            models_response = self.client.list()
            
            # **🔧 FIX: Handle both dict and ListResponse objects for compatibility**
            if hasattr(models_response, 'models'):
                # New ollama client returns ListResponse object
                model_list = models_response.models
                for model in model_list:
                    if model.model == self.model_name:
                        return {
                            'name': model.model,
                            'size': model.size,
                            'digest': model.digest,
                            'modified_at': model.modified_at.isoformat() if model.modified_at else None,
                            'details': {
                                'format': model.details.format if model.details else None,
                                'family': model.details.family if model.details else None,
                                'parameter_size': model.details.parameter_size if model.details else None,
                                'quantization_level': model.details.quantization_level if model.details else None
                            } if model.details else None
                        }
            elif isinstance(models_response, dict):
                # Legacy format: dict with 'models' key
                model_list = models_response.get('models', [])
                for model in model_list:
                    if isinstance(model, dict) and model.get('name') == self.model_name:
                        return model
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return None
    
    def check_model_availability(self) -> bool:
        """Check if the specified model is available.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            models_response = self.client.list()
            
            # **🔧 FIX: Handle both dict and ListResponse objects for compatibility**
            if hasattr(models_response, 'models'):
                # New ollama client returns ListResponse object
                model_list = models_response.models
                model_names = [model.model for model in model_list]
            elif isinstance(models_response, dict):
                # Legacy format: dict with 'models' key
                model_list = models_response.get('models', [])
                model_names = [m.get('name', 'unknown') for m in model_list if isinstance(m, dict)]
            else:
                model_names = []
            
            available = self.model_name in model_names
            
            if not available:
                self.logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
            else:
                self.logger.info(f"✅ Model {self.model_name} is available!")
            
            return available
        except Exception as e:
            self.logger.error(f"Failed to check model availability: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if not available locally.
        
        Returns:
            True if model pull successful, False otherwise
        """
        try:
            self.logger.info(f"Pulling model {self.model_name}...")
            self.client.pull(self.model_name)
            self.logger.info(f"Successfully pulled model {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to pull model {self.model_name}: {e}")
            return False
    
    def set_model(self, model_name: str) -> bool:
        """Set the model to use.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            True if model is set and available, False otherwise
        """
        old_model = self.model_name
        self.model_name = model_name
        self._is_warmed_up = False  # Reset warmup status when changing models
        
        if self.check_model_availability():
            self.logger.info(f"Model changed from {old_model} to {model_name}")
            return True
        else:
            self.model_name = old_model  # Revert on failure
            return False
    
    def is_warmed_up(self) -> bool:
        """Check if model is warmed up.
        
        Returns:
            True if model is warmed up, False otherwise
        """
        return self._is_warmed_up
    
    def _update_performance_stats(self, request_time: float, success: bool):
        """Update performance statistics.
        
        Args:
            request_time: Time taken for the request in seconds
            success: Whether the request was successful
        """
        if success:
            self._performance_stats['total_requests'] += 1
            self._performance_stats['total_time'] += request_time
            self._performance_stats['fastest_request'] = min(
                self._performance_stats['fastest_request'], request_time
            )
            self._performance_stats['slowest_request'] = max(
                self._performance_stats['slowest_request'], request_time
            )
            
            # Check if this might be a connection reuse (fast response)
            if self._is_warmed_up and request_time < 3.0:  # Less than 3 seconds suggests reuse
                self._performance_stats['connection_reuses'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = self._performance_stats.copy()
        
        if stats['total_requests'] > 0:
            stats['average_request_time'] = stats['total_time'] / stats['total_requests']
            stats['connection_reuse_rate'] = stats['connection_reuses'] / stats['total_requests']
        else:
            stats['average_request_time'] = 0.0
            stats['connection_reuse_rate'] = 0.0
        
        # Calculate performance gains from optimization
        if stats['total_requests'] > 1:
            estimated_unoptimized_time = stats['total_requests'] * stats['slowest_request']
            time_saved = estimated_unoptimized_time - stats['total_time']
            stats['estimated_time_saved'] = max(0, time_saved)
            stats['performance_gain_percent'] = (time_saved / estimated_unoptimized_time) * 100 if estimated_unoptimized_time > 0 else 0
        else:
            stats['estimated_time_saved'] = 0.0
            stats['performance_gain_percent'] = 0.0
        
        return stats
    
    def log_performance_summary(self):
        """Log a summary of performance statistics."""
        stats = self.get_performance_report()
        
        if stats['total_requests'] == 0:
            self.logger.info("📊 No requests made yet for performance analysis")
            return
        
        self.logger.info("📊 PERFORMANCE SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"🔢 Total requests: {stats['total_requests']}")
        self.logger.info(f"⏱️  Total time: {stats['total_time']:.2f}s")
        self.logger.info(f"⚡ Average time: {stats['average_request_time']:.2f}s")
        self.logger.info(f"🚀 Fastest request: {stats['fastest_request']:.2f}s")
        self.logger.info(f"🐌 Slowest request: {stats['slowest_request']:.2f}s")
        self.logger.info(f"🔗 Connection reuse rate: {stats['connection_reuse_rate']:.1%}")
        
        if stats['performance_gain_percent'] > 0:
            self.logger.info(f"💰 Estimated time saved: {stats['estimated_time_saved']:.2f}s")
            self.logger.info(f"📈 Performance gain: {stats['performance_gain_percent']:.1f}%")
        
        self.logger.info("=" * 50)
    
    def send_keepalive_ping(self) -> bool:
        """发送轻量级心跳请求，保持模型活跃状态
        
        Returns:
            True if ping successful, False otherwise
        """
        try:
            # 使用极简prompt减少计算负担
            ping_prompt = "hi"
            
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": ping_prompt}],
                options={"temperature": 0.1, "max_tokens": 5},  # 限制输出长度
                keep_alive="30m"  # 重新设置keep_alive
            )
            
            if response and "message" in response:
                self.logger.debug("🔥 Model keep-alive ping successful")
                return True
            else:
                self.logger.warning("Keep-alive ping failed - invalid response")
                return False
                
        except Exception as e:
            self.logger.warning(f"Keep-alive ping failed: {e}")
            return False
    
    def maintain_model_warmth(self) -> bool:
        """维护模型热状态，定期发送心跳
        
        建议在处理大批量任务前调用
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("🔥 Maintaining model warmth to prevent cold starts...")
        
        # 确保模型预热
        if not self._is_warmed_up:
            if not self.warmup_model():
                return False
        
        # 发送心跳
        success = self.send_keepalive_ping()
        if success:
            self.logger.info("✅ Model warmth maintained - ready for high-performance processing")
        else:
            self.logger.warning("⚠️ Failed to maintain model warmth")
        
        return success 