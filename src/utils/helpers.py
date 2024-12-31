# helpers.py

import numpy as np
from typing import List, Dict, Any, Union
import json
import time
from pathlib import Path

class EncryptionHelper:
    """
    Provides common utility methods for encryption tasks, such as
    formatting timestamps or converting bytes <-> hex strings.
    """

    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """
        Convert a numeric timestamp (Unix time) into a human-readable string.

        Args:
            timestamp: float representing Unix time

        Returns:
            A string in 'YYYY-MM-DD HH:MM:SS' format
        """
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

    @staticmethod
    def bytes_to_hex(data: bytes) -> str:
        """
        Convert raw bytes into a hexadecimal string.

        Args:
            data: bytes object

        Returns:
            Hexadecimal string representation of 'data'
        """
        return data.hex()

    @staticmethod
    def hex_to_bytes(hex_str: str) -> bytes:
        """
        Convert a hexadecimal string back into raw bytes.

        Args:
            hex_str: A string of hexadecimal characters

        Returns:
            The corresponding bytes object
        """
        return bytes.fromhex(hex_str)


class DataValidator:
    """
    DataValidator includes basic checks on seeds and messages,
    ensuring they meet expected formats or ranges.
    """

    @staticmethod
    def validate_seed(seed: Union[int, str, None]) -> bool:
        """
        Validate that 'seed' is within [0 .. 2^64-1], or None.

        Args:
            seed: integer, string convertible to int, or None

        Returns:
            True if valid, False otherwise
        """
        if seed is None:
            return True
        if not isinstance(seed, (int, str)):
            return False
        try:
            seed_int = int(seed)
            return 0 <= seed_int <= 2**64 - 1
        except ValueError:
            return False

    @staticmethod
    def validate_message(message: Union[str, bytes]) -> bool:
        """
        Validate that 'message' is not empty.

        Args:
            message: either a string or bytes

        Returns:
            True if non-empty, False otherwise
        """
        if isinstance(message, str):
            return len(message.encode()) > 0
        return len(message) > 0


class ConfigManager:
    """
    Loads, merges, and saves system configuration options, typically in JSON.

    Default fields:
      - max_attempts
      - entropy_threshold
      - min_message_length
      - max_message_length
      - visualization_options
      - performance_limits
    """

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.default_config = {
            "max_attempts": 100000,
            "entropy_threshold": 0.9999,
            "min_message_length": 1,
            "max_message_length": 1048576,  # 1MB
            "visualization_options": {
                "auto_refresh": True,
                "default_view": "combined",
                "plot_dpi": 100
            },
            "performance_limits": {
                "max_encryption_time": 1.0,
                "max_decryption_time": 1.0
            }
        }
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Attempt to load JSON configuration from self.config_path.
        If file not found or error, use default_config.

        Returns:
            A dictionary representing the loaded or merged config
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all required fields exist
                return {**self.default_config, **loaded_config}
            return self.default_config.copy()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config.copy()

    def save_config(self) -> bool:
        """
        Save current configuration (self.config) to self.config_path.

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value by 'key', or return 'default' if not found.

        Args:
            key: config field name
            default: fallback if key not present

        Returns:
            The corresponding config value, or default
        """
        return self.config.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        """
        Set a configuration value, stored in self.config.

        Args:
            key: config field name
            value: the new value
        """
        if key == "max_attempts" and (not isinstance(value, int) or value <= 0):
            raise ValueError("max_attempts must be a positive integer.")
        self.config[key] = value


class MetricsCollector:
    """
    Tracks performance metrics (encryption time, decryption time, entropies, etc.)
    and provides aggregated statistics or export capabilities.
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'encryption_times': [],
            'decryption_times': [],
            'entropy_values': [],
            'layer_counts': [],
            'timeline_depths': [],
            'message_sizes': []
        }
        self.start_time = time.time()

    def add_metric(self, category: str, value: float) -> None:
        """
        Add a numeric metric 'value' to the specified 'category' in metrics.

        Args:
            category: e.g. 'encryption_times', 'entropy_values', etc.
            value: numeric measurement
        """
        if category in self.metrics:
            self.metrics[category].append(value)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mean, std, min, max, count for each category, plus total time.

        Returns:
            A dict of categories -> stats
        """
        stats = {}
        for category, values in self.metrics.items():
            if values:
                stats[category] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'count': len(values),
                    'total_time': time.time() - self.start_time
                }
        return stats

    def export_to_file(self, path: Path) -> bool:
        """
        Export current stats to a JSON file at 'path'.

        Returns:
            True if success, False otherwise
        """
        try:
            stats = self.get_statistics()
            with open(path, 'w') as f:
                json.dump(stats, f, indent=4)
            return True
        except Exception as e:
            print(f"Error exporting metrics: {e}")
            return False

    def clear_metrics(self) -> None:
        """
        Clear all collected metrics and reset the start time.
        """
        for category in self.metrics:
            self.metrics[category].clear()
        self.start_time = time.time()

    def get_performance_summary(self) -> Dict[str, float]:
        """
        A quick summary of average times, average entropy, total messages, etc.

        Returns:
            A dict summarizing performance in a high-level manner
        """
        return {
            'avg_encryption_time': (np.mean(self.metrics['encryption_times'])
                if self.metrics['encryption_times'] else 0),
            'avg_decryption_time': (np.mean(self.metrics['decryption_times'])
                if self.metrics['decryption_times'] else 0),
            'avg_entropy': (np.mean(self.metrics['entropy_values'])
                if self.metrics['entropy_values'] else 0),
            'total_messages': len(self.metrics['message_sizes']),
            'total_runtime': time.time() - self.start_time
        }
