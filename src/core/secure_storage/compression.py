# src/core/secure_storage/compression.py

from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
import time
import base64
import logging
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class CompressionMetrics:
    """Tracks compression performance and statistics"""
    original_size: int = 0
    compressed_size: int = 0
    pattern_count: int = 0
    compression_ratio: float = 0.0
    processing_time: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    memory_used: int = 0

class LRUCache:
    """Least Recently Used Cache for pattern matching"""
    
    def __init__(self, maxsize: int = 5000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: bytes) -> Optional[int]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: bytes, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        self.cache[key] = value

class FastPatternMatcher:
    """Optimized pattern matching engine"""
    
    def __init__(self):
        self.cache = LRUCache()
        self.min_pattern_size = 2
        self.max_pattern_size = 256
        self.large_data_threshold = 500_000
        self.large_data_max_pattern = 64

    def find_patterns(self, data: bytes) -> Dict[bytes, int]:
        """Find repeated patterns in data"""
        length = len(data)
        if length > self.large_data_threshold:
            self.max_pattern_size = self.large_data_max_pattern
            logger.debug("Using reduced pattern size for large data")

        patterns = {}
        data_view = memoryview(data)
        max_len = min(self.max_pattern_size, (length // 2) + 1)

        for size in range(self.min_pattern_size, max_len):
            pos = 0
            while pos + size <= length:
                pattern = bytes(data_view[pos:pos + size])
                
                # Check cache first
                cached_count = self.cache.get(pattern)
                if cached_count is not None:
                    if cached_count > 1:
                        patterns[pattern] = cached_count
                    pos += 1
                    continue

                # Count occurrences
                count = 0
                next_pos = pos + size
                while True:
                    next_pos = data.find(pattern, next_pos)
                    if next_pos == -1:
                        break
                    count += 1
                    next_pos += size

                if count > 1:
                    patterns[pattern] = count
                    self.cache.put(pattern, count)

                pos += 1

        return patterns

class EnhancedCompressor:
    """Optimized compression implementation"""
    
    def __init__(self):
        self.pattern_matcher = FastPatternMatcher()
        self.patterns: List[bytes] = []
        self.pattern_to_id: Dict[bytes, int] = {}

    def compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data and return compressed bytes plus metadata"""
        start_time = time.perf_counter()

        # Clear out old patterns / maps
        self.patterns.clear()
        self.pattern_to_id.clear()

        # Find patterns
        patterns = self.pattern_matcher.find_patterns(data)
        if patterns:
            # Sort patterns by descending length
            sorted_patterns = sorted(patterns.keys(), key=lambda x: -len(x))

            # Build pattern -> ID mapping
            for pattern in sorted_patterns:
                if pattern not in self.pattern_to_id:
                    self.pattern_to_id[pattern] = len(self.patterns)
                    self.patterns.append(pattern)

            # Compress using patterns
            compressed = self._compress_with_patterns(data)
            compression_ratio = len(compressed) / len(data)
        else:
            compressed = data
            compression_ratio = 1.0

        processing_time = time.perf_counter() - start_time

        # Convert patterns to base64 for storage
        patterns_base64 = {
            str(idx): base64.b64encode(pattern).decode('utf-8')
            for pattern, idx in self.pattern_to_id.items()
        }

        return compressed, {
            'compressed': len(compressed) < len(data),
            'compression_ratio': compression_ratio,
            'patterns': patterns_base64,
            'processing_time': processing_time
        }

    def decompress(self, data: bytes) -> bytes:
        """Decompress data using stored patterns"""
        if not self.patterns:
            return data

        result = bytearray()
        pos = 0
        data_len = len(data)

        while pos < data_len:
            if data[pos] == 0xFF:  # Pattern marker
                if pos + 4 > data_len:
                    raise ValueError("Incomplete pattern marker in compressed data")
                
                pattern_length = data[pos + 1]
                pattern_id = int.from_bytes(data[pos + 2:pos + 4], 'big')
                
                if pattern_id >= len(self.patterns):
                    raise ValueError(f"Invalid pattern ID: {pattern_id}")
                    
                pattern = self.patterns[pattern_id]
                if len(pattern) != pattern_length:
                    raise ValueError(f"Pattern length mismatch for ID {pattern_id}")
                    
                result.extend(pattern)
                pos += 4
            else:
                result.append(data[pos])
                pos += 1

        return bytes(result)

    def _compress_with_patterns(self, data: bytes) -> bytes:
        """Replace patterns with references"""
        compressed = bytearray()
        pos = 0
        length = len(data)

        while pos < length:
            # Find longest matching pattern at current position
            match = None
            match_length = 0

            for pattern, pattern_id in self.pattern_to_id.items():
                pat_len = len(pattern)
                if (pat_len > match_length and 
                    pos + pat_len <= length and
                    data[pos:pos + pat_len] == pattern):
                    match = pattern
                    match_length = pat_len

            if match:
                pattern_id = self.pattern_to_id[match]
                # Write pattern marker: 0xFF + length + pattern_id (2 bytes)
                compressed.append(0xFF)
                compressed.append(len(match))
                compressed.extend(pattern_id.to_bytes(2, 'big'))
                pos += match_length
            else:
                compressed.append(data[pos])
                pos += 1

        return bytes(compressed)

    def set_patterns(self, patterns_dict: Dict[str, str]) -> None:
        """Set patterns from base64-encoded dictionary"""
        self.patterns.clear()
        # Convert string keys to integers and sort them
        sorted_keys = sorted(int(k) for k in patterns_dict.keys())
        for idx in sorted_keys:
            pattern_base64 = patterns_dict[str(idx)]
            pattern = base64.b64decode(pattern_base64)
            self.patterns.append(pattern)
            self.pattern_to_id[pattern] = idx

    def calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of the data"""
        if not data:
            return 0.0
            
        frequencies = np.bincount(np.frombuffer(data, dtype=np.uint8))
        probabilities = frequencies[frequencies > 0] / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)