# src/core/compress.py

from typing import Dict, List, Optional
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """
    Stores the results of a compression attempt, including:
      - compressed data
      - compression ratio (compressed / original size)
      - found patterns + pattern map
      - time taken to process
    """
    data: bytes
    compression_ratio: float
    patterns: List[bytes]
    pattern_map: Dict[bytes, int]
    processing_time: float


class LRUCache:
    """
    Least Recently Used (LRU) Cache using OrderedDict for accurate eviction.
    Stores pattern -> count mappings to speed up repeated pattern checks.
    """

    def __init__(self, maxsize: int = 5000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: bytes) -> Optional[int]:
        """
        Retrieve the count associated with 'key' if it exists in cache.
        Move key to the end to mark it recently used.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: bytes, value: int):
        """
        Insert or update the key -> value (pattern -> count) in LRU cache.
        Evict the least recently used item if over capacity.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)  # Remove LRU item
            self.cache[key] = value


class FastPatternMatcher:
    """
    Optimized pattern matching engine.
    Finds repeated byte patterns in data for compression.
    """

    def __init__(self):
        self.cache = LRUCache()
        self.min_pattern_size = 2
        self.max_pattern_size = 256

        # If data is large, we can skip or scale back pattern searching
        # to prevent huge performance hits. The user can tweak these.
        self.large_data_threshold = 500_000   # e.g. 500 KB
        self.large_data_max_pattern = 64      # max pattern size if data is "large"

    def find_patterns(self, data: bytes) -> Dict[bytes, int]:
        """
        Scan 'data' for repeated patterns of length in [min_pattern_size..max_pattern_size].
        Return a dict of pattern -> count (only if count>1).
        Large inputs are handled with limited or skipped searching to improve speed.
        """
        length = len(data)
        # If data is huge, limit or skip
        if length > self.large_data_threshold:
            # We still do minimal searching, but reduce max pattern size:
            print("Data is large; limiting pattern search for performance.")
            self.max_pattern_size = self.large_data_max_pattern

        patterns = {}
        data_view = memoryview(data)

        # The maximum pattern length is min(self.max_pattern_size, length//2+1).
        # This ensures we don't do unbounded searching for big data.
        max_len = min(self.max_pattern_size, (length // 2) + 1)

        for size in range(self.min_pattern_size, max_len):
            pos = 0
            while pos + size <= length:
                pattern = bytes(data_view[pos:pos + size])

                # Check cache first
                cached_count = self.cache.get(pattern)
                if cached_count is not None:
                    # If found repeated enough times
                    if cached_count > 1:
                        patterns[pattern] = cached_count
                    pos += 1
                    continue

                # Count occurrences of this pattern in the remainder
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
    """
    Main compression system that:
      1. Uses FastPatternMatcher to detect repeated patterns.
      2. Replaces patterns with references (0xFF marker + pattern_id) in final stream.
    """

    def __init__(self):
        self.pattern_matcher = FastPatternMatcher()
        self.patterns: List[bytes] = []       # Store discovered patterns for reference
        self.pattern_to_id: Dict[bytes, int] = {}  # Map pattern -> ID

    def compress(self, data: bytes) -> CompressionResult:
        start_time = time.perf_counter()

        # Clear out old patterns / maps
        self.patterns.clear()
        self.pattern_to_id.clear()

        # Step 1: find patterns
        patterns = self.pattern_matcher.find_patterns(data)
        if patterns:
            # Sort patterns by descending length so that we prioritize longer patterns
            sorted_patterns = sorted(patterns.keys(), key=lambda x: -len(x))

            # Build pattern -> ID mapping
            for pattern in sorted_patterns:
                if pattern not in self.pattern_to_id:
                    self.pattern_to_id[pattern] = len(self.patterns)
                    self.patterns.append(pattern)

            # Step 2: compress using those patterns
            compressed = self._compress_with_patterns(data)
            compression_ratio = len(compressed) / len(data)
        else:
            # No patterns found => no compression
            compressed = data
            compression_ratio = 1.0

        processing_time = time.perf_counter() - start_time
        return CompressionResult(
            data=compressed,
            compression_ratio=compression_ratio,
            patterns=self.patterns.copy(),
            pattern_map=self.pattern_to_id.copy(),
            processing_time=processing_time
        )

    def _compress_with_patterns(self, data: bytes) -> bytes:
        """
        Replace discovered patterns with:
          [0xFF][pattern_length][pattern_id_high][pattern_id_low]
        If no pattern matches, store literal byte.
        """
        compressed = bytearray()
        pos = 0
        length = len(data)

        while pos < length:
            match = None
            match_length = 0

            # Attempt to find the longest matching pattern at current pos
            for pattern in self.patterns:
                pat_len = len(pattern)
                if (pat_len > match_length and
                        pos + pat_len <= length and
                        data[pos:pos + pat_len] == pattern):
                    match = pattern
                    match_length = pat_len

            if match:
                pattern_id = self.pattern_to_id[match]
                compressed.append(0xFF)            # pattern marker
                compressed.append(len(match))      # pattern length
                compressed.append(pattern_id >> 8) # high byte
                compressed.append(pattern_id & 0xFF) # low byte
                pos += match_length
            else:
                # Store literal
                compressed.append(data[pos])
                pos += 1

        return bytes(compressed)

    def decompress(self, compressed: bytes) -> bytes:
        """
        Decompress the data back to original form using stored 'patterns'.
        Format for patterns: 0xFF + pattern_length + pattern_id(2 bytes)
        """
        decompressed = bytearray()
        pos = 0
        length = len(compressed)

        while pos < length:
            if compressed[pos] == 0xFF:
                # We expect at least 3 more bytes: length, id_high, id_low
                if pos + 3 >= length:
                    raise ValueError("Invalid compressed data format (missing pattern info).")

                pattern_length = compressed[pos + 1]
                pattern_id = (compressed[pos + 2] << 8) | compressed[pos + 3]

                if pattern_id >= len(self.patterns):
                    raise ValueError("Invalid pattern ID in compressed data.")

                pattern = self.patterns[pattern_id]
                # Double-check pattern length if needed
                if len(pattern) != pattern_length:
                    raise ValueError("Pattern length mismatch in compressed data.")

                decompressed.extend(pattern)
                pos += 4  # move past marker
            else:
                decompressed.append(compressed[pos])
                pos += 1

        return bytes(decompressed)

    def verify_compression(self, original: bytes, compressed: bytes, decompressed: bytes) -> bool:
        """
        Verify that decompressing 'compressed' yields the 'original'.
        """
        return original == decompressed
