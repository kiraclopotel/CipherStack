# src/core/secure_storage/storage.py

import json
import time
import hashlib
import base64
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from core.secure_storage.compression import EnhancedCompressor
from core.secure_storage.encryption import EncryptionManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BytesEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle bytes objects efficiently"""

    def default(self, obj):
        if isinstance(obj, bytes):
            return {'_bytes_': base64.b64encode(obj).decode('utf-8')}
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class SecureStorage:
    """
    Main storage system combining compression and encryption with optimized performance.
    Handles compression thresholds and efficient pattern storage.
    """

    def __init__(self):
        self.compressor = EnhancedCompressor()
        self.encryption_manager = EncryptionManager()
        self.metrics = {
            'compression': {},
            'encryption': {},
            'storage': {
                'total_operations': 0,
                'total_size': 0,
                'avg_compression_ratio': 0.0
            }
        }
        self.min_compress_size = 100  # Don't compress data smaller than this

    def save_state(self, state_data: Dict[str, Any], identifier: str) -> bytes:
        """
        Save state with optimized compression and encryption.
        Only compresses data above a threshold size.
        """
        try:
            # Convert state to JSON with minimal encoding
            state_bytes = json.dumps(state_data, cls=BytesEncoder, separators=(',', ':')).encode()
            logger.debug(f"Initial state size: {len(state_bytes)} bytes")

            # Calculate checksum on the original state_bytes
            original_checksum = hashlib.sha256(state_bytes).hexdigest()
            logger.debug(f"Original checksum (state_bytes): {original_checksum}")

            # Only compress if data is large enough
            if len(state_bytes) > self.min_compress_size:
                compressed = self.compressor.compress(state_bytes)
                should_compress = compressed[1]['compressed'] and compressed[0] is not None
            else:
                should_compress = False

            if should_compress:
                logger.debug(f"Compressing data: {len(state_bytes)} -> {len(compressed[0])} bytes")
                data_to_encrypt = compressed[0]
                compression_metadata = compressed[1]
            else:
                logger.debug("Skipping compression - would not reduce size")
                data_to_encrypt = state_bytes
                compression_metadata = {'compressed': False}

            # Encrypt the data
            encrypted_data = self.encryption_manager.encrypt_data(data_to_encrypt, identifier)

            # Create final data structure
            final_data = {
                'v': '3.0',
                't': int(time.time()),
                'e': base64.b64encode(encrypted_data['ciphertext']).decode('utf-8'),
                'm': encrypted_data['metadata'],
                'c': compression_metadata['compressed'],
                'h': original_checksum
            }

            if compression_metadata['compressed']:
                final_data['patterns'] = compression_metadata.get('patterns', {})

            # Convert final structure to bytes
            final_bytes = json.dumps(final_data, cls=BytesEncoder, separators=(',', ':')).encode()

            # Update metrics
            self._update_metrics(
                original_size=len(state_bytes),
                encrypted_size=len(data_to_encrypt),
                compression_metadata=compression_metadata,
                encryption_metadata=encrypted_data['metadata']
            )

            logger.debug(f"State saved successfully. Final size: {len(final_bytes)} bytes")
            return final_bytes

        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            raise

    def load_state(self, enc_data: bytes, identifier: str) -> Optional[Dict[str, Any]]:
            """Load state with proper byte handling"""
            try:
                # Parse the enc file data
                data = json.loads(enc_data.decode())
                logger.debug(f"Loaded data structure: {data}")

                # Extract components
                try:
                    ciphertext_b64 = data['e']
                    if isinstance(ciphertext_b64, dict) and '_bytes_' in ciphertext_b64:
                        ciphertext = base64.b64decode(ciphertext_b64['_bytes_'])
                    elif isinstance(ciphertext_b64, str):
                        ciphertext = base64.b64decode(ciphertext_b64)
                    else:
                        ciphertext = ciphertext_b64

                    metadata = data['m']
                    logger.debug(f"Extracted ciphertext and metadata.")

                except (KeyError, ValueError, TypeError) as e:
                    logger.error(f"Invalid data format: {e}")
                    return None

                # Prepare for decryption
                encrypted_data = {
                    'ciphertext': ciphertext,
                    'metadata': metadata
                }

                # Decrypt
                decrypted_data = self.encryption_manager.decrypt_data(encrypted_data, identifier)
                if not decrypted_data:
                    logger.error("Decryption failed")
                    return None

                logger.debug(f"Decrypted data size: {len(decrypted_data)} bytes")

                # Handle compression
                if data.get('c', False):
                    try:
                        # Load patterns before decompression
                        patterns_dict = {}
                        if 'patterns' in data:
                            for pattern_id, pattern_data in data['patterns'].items():
                                if isinstance(pattern_data, dict) and '_bytes_' in pattern_data:
                                    patterns_dict[pattern_id] = pattern_data['_bytes_']
                                else:
                                    patterns_dict[pattern_id] = pattern_data
                                    
                        # Set patterns in compressor
                        self.compressor.set_patterns(patterns_dict)
                        
                        # Now decompress
                        decompressed_data = self.compressor.decompress(decrypted_data)
                        logger.debug("Data decompressed successfully.")
                    except Exception as e:
                        logger.error(f"Decompression failed: {str(e)}")
                        return None
                else:
                    decompressed_data = decrypted_data
                    logger.debug("Data was not compressed.")

                # Verify checksum
                calculated_checksum = hashlib.sha256(decompressed_data).hexdigest()
                expected_checksum = data.get('h', '')
                if calculated_checksum != expected_checksum:
                    logger.error("Checksum verification failed")
                    return None

                # Parse state data
                try:
                    state_data = json.loads(decompressed_data.decode('utf-8'))
                    logger.debug("State data parsed successfully.")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode failed: {e}")
                    return None

                # Convert special objects
                self._convert_bytes_objects(state_data)

                logger.debug("State loaded and verified successfully.")
                return state_data

            except Exception as e:
                logger.error(f"Error loading state: {str(e)}")
                return None

    def _convert_bytes_objects(self, data: Any) -> None:
        """Recursively convert special byte objects with improved error handling"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    if '_bytes_' in value:
                        try:
                            decoded = self._safe_decode_base64(value['_bytes_'])
                            if decoded is not None:
                                data[key] = decoded
                        except Exception as e:
                            logger.error(f"Failed to decode bytes for key {key}: {e}")
                    else:
                        self._convert_bytes_objects(value)
                elif isinstance(value, list):
                    self._convert_bytes_objects(value)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    if '_bytes_' in item:
                        try:
                            decoded = self._safe_decode_base64(item['_bytes_'])
                            if decoded is not None:
                                data[i] = decoded
                        except Exception as e:
                            logger.error(f"Failed to decode bytes at index {i}: {e}")
                    else:
                        self._convert_bytes_objects(item)
                elif isinstance(item, list):
                    self._convert_bytes_objects(item)

    def _safe_decode_base64(self, data: str) -> Optional[bytes]:
        """Safely decode base64 data with error handling"""
        try:
            return base64.b64decode(data)
        except Exception as e:
            logger.error(f"Base64 decode failed: {str(e)}")
            return None

    def _update_metrics(self, original_size: int, encrypted_size: int,
                       compression_metadata: Dict[str, Any],
                       encryption_metadata: Dict[str, Any]) -> None:
        """Update storage metrics with basic validation"""
        if original_size > 0 and encrypted_size > 0:
            self.metrics['compression'] = compression_metadata.get('metrics', {})
            self.metrics['encryption'] = encryption_metadata.get('metrics', {})

            self.metrics['storage']['total_operations'] += 1
            self.metrics['storage']['total_size'] += original_size

            ratio = encrypted_size / original_size
            total_ops = self.metrics['storage']['total_operations']
            current_avg = self.metrics['storage']['avg_compression_ratio']

            self.metrics['storage']['avg_compression_ratio'] = (
                (current_avg * (total_ops - 1) + ratio) / total_ops
            )
            logger.debug(f"Updated metrics: {self.metrics}")


class FileHandler:
    """Handles file operations with proper error checking"""

    @staticmethod
    def save_enc_file(data: bytes, filepath: str) -> bool:
        """Save encrypted data to file with validation"""
        try:
            if not data:
                logger.error("Cannot save empty data")
                return False

            path = Path(filepath)
            if path.exists():
                logger.warning(f"Overwriting existing file: {filepath}")

            with open(filepath, 'wb') as f:
                f.write(data)

            # Verify file was written correctly
            if not path.exists() or path.stat().st_size != len(data):
                logger.error("File verification failed after write")
                return False

            logger.debug(f"Successfully saved {len(data)} bytes to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving file {filepath}: {str(e)}")
            return False

    @staticmethod
    def load_enc_file(filepath: str) -> Optional[bytes]:
        """Load encrypted data with basic validation"""
        try:
            path = Path(filepath)
            if not path.exists():
                logger.error(f"File not found: {filepath}")
                return None

            if path.stat().st_size == 0:
                logger.error(f"File is empty: {filepath}")
                return None

            with open(filepath, 'rb') as f:
                data = f.read()

            logger.debug(f"Successfully loaded {len(data)} bytes from {filepath}")
            return data

        except Exception as e:
            logger.error(f"Error loading file {filepath}: {str(e)}")
            return None


def create_secure_storage() -> SecureStorage:
    """Create and return a configured SecureStorage instance"""
    return SecureStorage()