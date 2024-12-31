# src/core/secure_storage/encryption.py

from dataclasses import dataclass
import hashlib
import time
import logging
import base64
from typing import Dict, Any, Optional, Tuple
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class SecurityMetrics:
    """Tracks security-related metrics"""
    encryption_time: float
    key_derivation_time: float
    total_operations: int
    hash_verification: bool

class SecureAESEncryptor:
    """Enhanced AES encryption implementation"""
    
    def __init__(self):
        self.key_iterations = 200000
        self.key_size = 32  # AES-256
        self.salt_size = 32
        self.metrics = SecurityMetrics(
            encryption_time=0.0,
            key_derivation_time=0.0,
            total_operations=0,
            hash_verification=False
        )

    def encrypt(self, data: bytes, key: bytes) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt data using AES-GCM
        Returns (nonce, tag, ciphertext)
        """
        start_time = time.perf_counter()
        cipher = AES.new(key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(data)
        
        self.metrics.encryption_time = time.perf_counter() - start_time
        self.metrics.total_operations += 1
        
        return cipher.nonce, tag, ciphertext

    def decrypt(self, data: bytes, key: bytes, nonce: bytes, tag: bytes) -> bytes:
        """Decrypt data using AES-GCM"""
        start_time = time.perf_counter()
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        decrypted = cipher.decrypt_and_verify(data, tag)
        
        self.metrics.encryption_time = time.perf_counter() - start_time
        self.metrics.total_operations += 1
        
        return decrypted

    def derive_key(self, identifier: str, salt: bytes) -> bytes:
        """Derive encryption key from identifier and salt"""
        start_time = time.perf_counter()
        key = PBKDF2(
            identifier.encode(),
            salt,
            dkLen=self.key_size,
            count=self.key_iterations
        )
        self.metrics.key_derivation_time = time.perf_counter() - start_time
        return key

    def generate_salt(self) -> bytes:
        """Generate a random salt"""
        return get_random_bytes(self.salt_size)


class EncryptionManager:
    """Handles high-level encryption operations"""
    
    def __init__(self):
        self.key_iterations = 200000
        self.key_size = 32  # AES-256
        self.salt_size = 32
        self.block_size = AES.block_size

    def encrypt_data(self, data: bytes, identifier: str) -> Dict[str, Any]:
        """Encrypt data using consistent AES-CBC mode"""
        try:
            # Generate salt and derive key
            salt = get_random_bytes(self.salt_size)
            key = PBKDF2(
                identifier.encode(),
                salt,
                dkLen=self.key_size,
                count=self.key_iterations
            )

            # Create cipher and encrypt
            cipher = AES.new(key, AES.MODE_CBC)
            padded_data = pad(data, AES.block_size)
            ciphertext = cipher.encrypt(padded_data)

            # Store metadata including IV
            metadata = {
                'salt': base64.b64encode(salt).decode('utf-8'),
                'iv': base64.b64encode(cipher.iv).decode('utf-8'),  # Changed 'nonce' to 'iv' for clarity
                'mode': 'CBC',
                'original_length': len(data)
                # Removed 'tag' as AES-CBC does not use tags
            }

            logger.debug(f"Encryption successful. Metadata: {metadata}")

            return {
                'ciphertext': ciphertext,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise

    def decrypt_data(self, encrypted_data: Dict[str, Any], identifier: str) -> Optional[bytes]:
        """Decrypt data using consistent AES-CBC mode"""
        try:
            # Extract components
            ciphertext = encrypted_data['ciphertext']
            salt = base64.b64decode(encrypted_data['metadata']['salt'])
            iv = base64.b64decode(encrypted_data['metadata']['iv'])  # Changed 'nonce' to 'iv'

            # Derive key
            key = PBKDF2(
                identifier.encode(),
                salt,
                dkLen=self.key_size,
                count=self.key_iterations
            )

            # Create cipher and decrypt
            cipher = AES.new(key, AES.MODE_CBC, iv)
            padded_plaintext = cipher.decrypt(ciphertext)

            try:
                plaintext = unpad(padded_plaintext, AES.block_size)
                # Verify length if provided
                if 'original_length' in encrypted_data['metadata']:
                    if len(plaintext) != encrypted_data['metadata']['original_length']:
                        logger.error("Length verification failed")
                        return None
                logger.debug("Decryption successful.")
                return plaintext

            except ValueError as ve:
                logger.error(f"Padding error during decryption: {ve}")
                return None

        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return None

    def derive_key(self, identifier: str, salt: bytes) -> bytes:
        """Derive encryption key from identifier and salt"""
        return PBKDF2(
            identifier.encode(),
            salt,
            dkLen=self.key_size,
            count=self.key_iterations
        )