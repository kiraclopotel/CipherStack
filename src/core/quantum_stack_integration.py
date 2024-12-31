# src/core/quantum_stack_integration.py

import base64
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from core.secure_storage.storage import SecureStorage, FileHandler, create_secure_storage
from core.encryption import QuantumStackEncryption

class MessageEncryption:
    """Individual message encryption handler"""
    def __init__(self, message: bytes, seed: int, keyword: str):
        self.message = message
        self.seed = seed
        self.keyword = keyword
        self.encryptor = QuantumStackEncryption()
        
        # Create unique key from seed + keyword
        self.unique_key = hashlib.sha256(
            f"{seed}:{keyword}".encode()
        ).digest()

    def encrypt(self) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Encrypt message with unique key"""
        try:
            # Add additional entropy from unique key
            enhanced_seed = int.from_bytes(
                hashlib.sha256(self.unique_key + str(self.seed).encode()).digest()[:8],
                'big'
            )
            
            # Encrypt message
            success, entropy = self.encryptor.add_message(self.message, seed=enhanced_seed)
            if not success:
                return None, False
                
            # Get encryption data
            if not self.encryptor.encryption_data:
                return None, False
                
            iv, ciphertext, stored_entropy, coeffs = self.encryptor.encryption_data[-1]
            
            # Create verification hash
            verification = hashlib.sha256(
                self.unique_key + self.message
            ).hexdigest()
            
            return {
                'iv': base64.b64encode(iv).decode('utf-8'),
                'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
                'entropy': stored_entropy,
                'coefficients': [float(c) for c in coeffs] if coeffs else None,
                'verification': verification,
                'seed_hash': hashlib.sha256(str(self.seed).encode()).hexdigest()
            }, True
            
        except Exception as e:
            print(f"Encryption error: {str(e)}")
            return None, False

class QuantumStackIntegrator:
    def __init__(self):
        self.secure_storage = create_secure_storage()
        self.file_handler = FileHandler()

    def encrypt_messages_for_recipients(
        self,
        messages: List[bytes],
        recipient_data: Dict[str, Tuple[int, str]],
        identifier_length: int = 15
    ) -> Optional[str]:
        try:
            recipient_storage = {}
            
            # Encrypt each message independently
            for recipient, (seed, keyword) in recipient_data.items():
                msg_idx = list(recipient_data.keys()).index(recipient)
                if msg_idx >= len(messages):
                    continue
                    
                message = messages[msg_idx]
                
                # Create isolated encryption for this message
                encryptor = MessageEncryption(message, seed, keyword)
                encrypted_data, success = encryptor.encrypt()
                
                if not success or not encrypted_data:
                    print(f"Failed to encrypt message for recipient {recipient}")
                    continue
                
                # Store with minimal seed info
                recipient_storage[recipient] = {
                    **encrypted_data,
                    'timestamp': int(time.time()),
                    'message_index': msg_idx
                }
            
            if not recipient_storage:
                return None
                
            state_data = {
                'version': '3.0',
                'recipients': recipient_storage,
                'timestamp': int(time.time())
            }
            
            # Generate identifier
            identifier = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:identifier_length]
            
            # Save state
            enc_data = self.secure_storage.save_state(state_data, identifier)
            filename = f"quantum_stack_{int(time.time())}.enc"
            
            if not self.file_handler.save_enc_file(enc_data, filename):
                return None
            
            return identifier
            
        except Exception as e:
            print(f"Error in message encryption: {str(e)}")
            return None

    def decrypt_message(
        self,
        identifier: str,
        recipient: str,
        seed: int,
        keyword: str,
        enc_file: str
    ) -> Optional[bytes]:
        try:
            # Load encrypted data
            enc_data = self.file_handler.load_enc_file(enc_file)
            if not enc_data:
                return None

            # Load state
            state = self.secure_storage.load_state(enc_data, identifier)
            if not state:
                return None
            
            # Get recipient's data SILENTLY - don't log anything about other recipients
            recipient_data = state.get('recipients', {}).get(recipient)
            if not recipient_data:
                return None
            
            # Verify seed matches (through hash)
            seed_hash = hashlib.sha256(str(seed).encode()).hexdigest()
            if seed_hash != recipient_data.get('seed_hash'):
                return None
            
            # Create unique key for verification
            unique_key = hashlib.sha256(
                f"{seed}:{keyword}".encode()
            ).digest()
            
            try:
                iv = base64.b64decode(recipient_data['iv'])
                ciphertext = base64.b64decode(recipient_data['ciphertext'])
                coeffs = recipient_data.get('coefficients')
                
                # Create decryptor with enhanced seed
                decryptor = QuantumStackEncryption()
                enhanced_seed = int.from_bytes(
                    hashlib.sha256(unique_key + str(seed).encode()).digest()[:8],
                    'big'
                )
                
                # Decrypt
                try:
                    if coeffs:
                        message = decryptor.decrypt_with_seed_and_coeffs(
                            ciphertext, enhanced_seed, iv, coeffs
                        )
                    else:
                        message = decryptor.decrypt_with_seed(
                            ciphertext, enhanced_seed, iv
                        )
                    
                    if not message:
                        return None
                        
                    # Verify message
                    verification = hashlib.sha256(
                        unique_key + message
                    ).hexdigest()
                    
                    if verification != recipient_data.get('verification'):
                        return None
                        
                    return message
                    
                except Exception:
                    return None
                    
            except Exception:
                return None

        except Exception:
            return None