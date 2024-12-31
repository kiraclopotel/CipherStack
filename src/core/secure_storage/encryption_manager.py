# encryption_manager.py

import base64
import hashlib
import logging
import os
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EncryptionManager:
    """Handles high-level encryption operations"""
    
    def __init__(self):
        self.key_iterations = 200000
        self.key_size = 32
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
                'nonce': base64.b64encode(cipher.iv).decode('utf-8'),
                'mode': 'CBC',
                'original_length': len(data)
                # Removed 'tag' as AES-CBC does not use tags
            }

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
            iv = base64.b64decode(encrypted_data['metadata']['nonce'])

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
