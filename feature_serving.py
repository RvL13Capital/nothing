# feature_serving.py
"""
High-Performance Feature Serving Infrastructure
Sub-millisecond feature retrieval for real-time inference
"""

import asyncio
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle
import lz4.frame
import msgpack

# Redis Cluster for distributed caching
import redis.asyncio as redis
from redis.asyncio.cluster import RedisCluster

# Apache Kafka for streaming
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# Vector database for similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FeatureRequest:
    """Feature serving request"""
    ticker: str
    feature_names: List[str]
    timestamp: Optional[datetime] = None
    max_age_seconds: int = 3600
    include_metadata: bool = False
    request_id: str = None

@dataclass
class FeatureResponse:
    """Feature serving response"""
    ticker: str
    features: Dict[str, Any]
    timestamp: datetime
    cache_hit: bool
    latency_ms: float
    metadata: Dict[str, Any] = None
    request_id: str = None

class FeatureCache:
    """Multi-tier feature caching system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # L1 Cache: In-memory (fastest)
        self.l1_cache = {}
        self.l1_max_size = config.get('l1_max_size', 10000)
        
        # L2 Cache: Redis Cluster (fast)
        self.redis_cluster = None
        
        # L3 Cache: Compressed storage (efficient)
        self.compression_enabled = config.get('compression', True)
        
        # Cache statistics
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0,
            'total_requests': 0
        }
    
    async def initialize(self):
        """Initialize cache infrastructure"""
        
        # Initialize Redis Cluster
        redis_config = self.config.get('redis_cluster', {})
        if redis_config:
            startup_nodes = redis_config.get('nodes', [{'host': 'localhost', 'port': 6379}])
            self.redis_cluster = RedisCluster(
                startup_nodes=startup_nodes,
                password=redis_config.get('password'),
                decode_responses=False,
                skip_full_coverage_check=True
            )
            
        logger.info("Feature cache initialized with multi-tier architecture")
    
    async def get(self, cache_key: str) -> Optional[Any]:
        """Get from cache with L1 -> L2 -> L3 fallback"""
        
        self.stats['total_requests'] += 1
        
        # L1 Cache check (in-memory)
        if cache_key in self.l1_cache:
            self.stats['l1_hits'] += 1
            return self.l1_cache[cache_key]
        
        self.stats['l1_misses'] += 1
        
        # L2 Cache check (Redis)
        if self.redis_cluster:
            try:
                cached_data = await self.redis_cluster.get(cache_key)
                if cached_data:
                    self.stats['l2_hits'] += 1
                    
                    # Deserialize
                    if self.compression_enabled:
                        decompressed = lz4.frame.decompress(cached_data)
                        data = msgpack.unpackb(decompressed, raw=False)
                    else:
                        data = pickle.loads(cached_data)
                    
                    # Promote to L1 cache
                    await self._promote_to_l1(cache_key, data)
                    return data
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        self.stats['l2_misses'] += 1
        return None
    
    async def set(self, cache_key: str, data: Any, ttl_seconds: int = 3600):
        """Set in cache with L1 and L2 storage"""
        
        # Store in L1 cache
        await self._promote_to_l1(cache_key, data)
        
        # Store in L2 cache (Redis)
        if self.redis_cluster:
            try:
                if self.compression_enabled:
                    packed = msgpack.packb(data, use_bin_type=True)
                    compressed = lz4.frame.compress(packed)
                    await self.redis_cluster.setex(cache_key, ttl_seconds, compressed)
                else:
                    serialized = pickle.dumps(data)
                    await self.redis_cluster.setex(cache_key, ttl_seconds, serialized)
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
    
    async def _promote_to_l1(self, cache_key: str, data: Any):
        """Promote data to L1 cache with LRU eviction"""
        
        if len(self.l1_cache) >= self.l1_max_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
        
        self.l1_cache[cache_key] = data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_l1 = self.stats['l1_hits'] + self.stats['l1_misses']
        total_l2 = self.stats['l2_hits'] + self.stats['l2_misses']
        
        return {
            'l1_hit_rate': self.stats['l1_hits'] / total_l1 if total_l1 > 0 else 0,
            'l2_hit_rate': self.stats['l2_hits'] / total_l2 if total_l2 > 0 else 0,
            'overall_hit_rate': (self.stats['l1_hits'] + self.stats['l2_hits']) / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0,
            **self.stats
        }

class FeatureStreamProcessor:
    """Real-time feature streaming with Kafka"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.producer = None
        self.consumers = {}
        
        self.kafka_config = config.get('kafka', {
            'bootstrap_servers': ['localhost:9092'],
            'feature_topic': 'market_features',
            'prediction_topic': 'breakout_predictions'
        })
    
    async def initialize(self):
        """Initialize Kafka producer and consumers"""
        
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available - streaming features disabled")
            return
        
        try:
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                value_serializer=lambda v: msgpack.packb(v, use_bin_type=True),
                compression_type='lz4'
            )
            
            logger.info("Kafka producer initialized for feature streaming")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
    
    async def stream_features(self, ticker: str, features: Dict[str, Any]):
        """Stream computed features to Kafka topic"""
        
        if not self.producer:
            return
        
        try:
            message = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'features': features,
                'message_type': 'feature_update'
            }
            
            future = self.producer.send(
                self.kafka_config['feature_topic'],
                key=ticker.encode('utf-8'),
                value=message
            )
            
            # Non-blocking send
            self.producer.flush(timeout=0.1)
            
        except Exception as e:
            logger.error(f"Failed to stream features for {ticker}: {e}")
    
    async def stream_prediction(self, ticker: str, prediction: Dict[str, Any]):
        """Stream breakout predictions to Kafka topic"""
        
        if not self.producer:
            return
        
        try:
            message = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'message_type': 'breakout_prediction'
            }
            
            future = self.producer.send(
                self.kafka_config['prediction_topic'],
                key=ticker.encode('utf-8'),
                value=message
            )
            
            self.producer.flush(timeout=0.1)
            
        except Exception as e:
            logger.error(f"Failed to stream prediction for {ticker}: {e}")

class FeatureVectorStore:
    """Vector store for feature similarity search"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index = None
        self.feature_metadata = {}
        self.dimension = config.get('vector_dimension', 128)
        
    def initialize(self):
        """Initialize FAISS vector index"""
        
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - vector search disabled")
            return
        
        # Initialize FAISS index for similarity search
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
        logger.info(f"Vector store initialized with dimension {self.dimension}")
    
    def add_feature_vector(self, ticker: str, features: np.ndarray, metadata: Dict[str, Any]):
        """Add feature vector to the index"""
        
        if self.index is None or len(features) != self.dimension:
            return
        
        # Normalize vector
        feature_vector = features.reshape(1, -1).astype('float32')
        faiss.normalize_L2(feature_vector)
        
        # Add to index
        vector_id = self.index.ntotal
        self.index.add(feature_vector)
        
        # Store metadata
        self.feature_metadata[vector_id] = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            **metadata
        }
    
    def find_similar_patterns(self, query_features: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Find similar feature patterns"""
        
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_vector = query_features.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search for similar vectors
        similarities, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx in self.feature_metadata:
                result = {
                    'similarity': float(similarity),
                    'rank': i + 1,
                    **self.feature_metadata[idx]
                }
                results.append(result)
        
        return results

class EnterpriseFeatureServer:
    """High-performance feature serving system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.cache = FeatureCache(config.get('cache', {}))
        self.stream_processor = FeatureStreamProcessor(config.get('streaming', {}))
        self.vector_store = FeatureVectorStore(config.get('vector_store', {}))
        
        # Performance monitoring
        self.request_latencies = []
        self.throughput_counter = 0
        self.last_throughput_reset = datetime.now()
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 20))
    
    async def initialize(self):
        """Initialize all serving components"""
        await self.cache.initialize()
        await self.stream_processor.initialize()
        self.vector_store.initialize()
        
        logger.info("Enterprise Feature Server initialized successfully")
    
    async def serve_features(self, request: FeatureRequest) -> FeatureResponse:
        """Serve features with sub-millisecond latency"""
        
        start_time = datetime.now()
        request_id = request.request_id or self._generate_request_id()
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try cache first
        cached_features = await self.cache.get(cache_key)
        
        if cached_features and self._is_cache_valid(cached_features, request.max_age_seconds):
            # Cache hit
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.request_latencies.append(latency_ms)
            
            return FeatureResponse(
                ticker=request.ticker,
                features=cached_features['features'],
                timestamp=cached_features['timestamp'],
                cache_hit=True,
                latency_ms=latency_ms,
                metadata=cached_features.get('metadata') if request.include_metadata else None,
                request_id=request_id
            )
        
        # Cache miss - compute or retrieve features
        features_data = await self._compute_or_retrieve_features(request)
        
        if features_data:
            # Cache the result
            cache_data = {
                'features': features_data,
                'timestamp': datetime.now(),
                'metadata': {'computed': True, 'request_id': request_id}
            }
            
            await self.cache.set(cache_key, cache_data, request.max_age_seconds)
            
            # Stream to Kafka if enabled
            await self.stream_processor.stream_features(request.ticker, features_data)
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.request_latencies.append(latency_ms)
        self.throughput_counter += 1
        
        return FeatureResponse(
            ticker=request.ticker,
            features=features_data or {},
            timestamp=datetime.now(),
            cache_hit=False,
            latency_ms=latency_ms,
            metadata={'computed': True} if request.include_metadata else None,
            request_id=request_id
        )
    
    async def serve_batch_features(self, requests: List[FeatureRequest]) -> List[FeatureResponse]:
        """Serve multiple feature requests in parallel"""
        
        # Process requests concurrently
        tasks = [self.serve_features(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Error processing request {i}: {response}")
                # Create error response
                valid_responses.append(FeatureResponse(
                    ticker=requests[i].ticker,
                    features={},
                    timestamp=datetime.now(),
                    cache_hit=False,
                    latency_ms=0,
                    request_id=requests[i].request_id
                ))
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    async def _compute_or_retrieve_features(self, request: FeatureRequest) -> Optional[Dict[str, Any]]:
        """Compute or retrieve features from storage"""
        
        # This would integrate with the feature store
        # For now, return mock data
        return {
            feature: np.random.random() for feature in request.feature_names
        }
    
    def _generate_cache_key(self, request: FeatureRequest) -> str:
        """Generate deterministic cache key"""
        key_data = f"{request.ticker}:{':'.join(sorted(request.feature_names))}"
        if request.timestamp:
            key_data += f":{request.timestamp.isoformat()}"
        
        return f"features:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return hashlib.md5(f"{datetime.now().isoformat()}:{np.random.random()}".encode()).hexdigest()[:8]
    
    def _is_cache_valid(self, cached_data: Dict[str, Any], max_age_seconds: int) -> bool:
        """Check if cached data is still valid"""
        if 'timestamp' not in cached_data:
            return False
        
        cache_age = (datetime.now() - cached_data['timestamp']).total_seconds()
        return cache_age <= max_age_seconds
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get serving performance metrics"""
        
        now = datetime.now()
        time_delta = (now - self.last_throughput_reset).total_seconds()
        
        metrics = {
            'avg_latency_ms': np.mean(self.request_latencies) if self.request_latencies else 0,
            'p95_latency_ms': np.percentile(self.request_latencies, 95) if self.request_latencies else 0,
            'p99_latency_ms': np.percentile(self.request_latencies, 99) if self.request_latencies else 0,
            'throughput_rps': self.throughput_counter / time_delta if time_delta > 0 else 0,
            'total_requests': len(self.request_latencies),
            'cache_stats': self.cache.get_stats()
        }
        
        # Reset counters periodically
        if time_delta > 300:  # 5 minutes
            self.request_latencies = self.request_latencies[-1000:]  # Keep last 1000
            self.throughput_counter = 0
            self.last_throughput_reset = now
        
        return metrics

# Usage example and integration
async def setup_enterprise_feature_serving(config: Dict[str, Any]) -> EnterpriseFeatureServer:
    """Setup enterprise feature serving infrastructure"""
    
    feature_server = EnterpriseFeatureServer(config)
    await feature_server.initialize()
    
    logger.info("Enterprise feature serving ready for production traffic")
    return feature_server

# FastAPI integration for REST API
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List
    
    class FeatureRequestModel(BaseModel):
        ticker: str
        feature_names: List[str]
        max_age_seconds: int = 3600
        include_metadata: bool = False
    
    def create_feature_serving_api(feature_server: EnterpriseFeatureServer) -> FastAPI:
        """Create FastAPI app for feature serving"""
        
        app = FastAPI(title="Enterprise Feature Serving API", version="2.0.0")
        
        @app.post("/features", response_model=dict)
        async def serve_features(request: FeatureRequestModel):
            """Serve features for a single ticker"""
            
            feature_request = FeatureRequest(
                ticker=request.ticker,
                feature_names=request.feature_names,
                max_age_seconds=request.max_age_seconds,
                include_metadata=request.include_metadata
            )
            
            response = await feature_server.serve_features(feature_request)
            return asdict(response)
        
        @app.post("/features/batch", response_model=List[dict])
        async def serve_batch_features(requests: List[FeatureRequestModel]):
            """Serve features for multiple tickers"""
            
            feature_requests = [
                FeatureRequest(
                    ticker=req.ticker,
                    feature_names=req.feature_names,
                    max_age_seconds=req.max_age_seconds,
                    include_metadata=req.include_metadata
                ) for req in requests
            ]
            
            responses = await feature_server.serve_batch_features(feature_requests)
            return [asdict(response) for response in responses]
        
        @app.get("/metrics")
        async def get_metrics():
            """Get performance metrics"""
            return feature_server.get_performance_metrics()
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        return app

except ImportError:
    logger.warning("FastAPI not available - REST API disabled")