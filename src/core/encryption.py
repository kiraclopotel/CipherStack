# src/core/encryption.py

from utils.helpers import ConfigManager
import numpy as np
from numpy.random import default_rng
from scipy.special import erfc
from scipy.stats import chi2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from typing import Tuple, Optional, List, Dict, Any
import struct
import hashlib
import random
import pickle
from pathlib import Path
import logging

from core.mathematics import MathematicalCore
from core.timeline import TimelineManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# c_k advanced logic
# --------------------------------------------------------------------------
def generate_coefficients(count: int = 100) -> List[float]:
    """
    Generate `count` random float coefficients in the range (-1, 1).
    Used for advanced c_k usage.
    """
    coeffs = [(random.random() * 2.0) - 1.0 for _ in range(count)]
    logger.debug(f"Generated coefficients: {coeffs}")
    return coeffs

def combine_key_with_coeffs(base_key: bytes, coeffs: List[float]) -> bytes:
    """
    Combine a 32-byte AES key with c_k coefficients:
     1) Convert `coeffs` to a stable string (like 0.123456)
     2) Hash => 32 bytes
     3) XOR with base_key => final key
    """
    text = ''.join(f"{c:.6f}" for c in coeffs).encode('utf-8')
    c_hash = hashlib.sha256(text).digest()
    final_key = bytes(a ^ b for a, b in zip(base_key, c_hash))
    logger.debug(f"Combined key with coefficients. Final key: {final_key.hex()}")
    return final_key

class QuantumStackEncryption:
    """
    Encryption system that supports:
      - Basic AES encryption with seed (custom or random)
      - Optional advanced c_k usage if requested
      - Aggregator logic for combining/extracting messages (with or without c_k)
      - Timeline integration for each stored message
      - Perfect entropy seed searching in [1..2^32) if user seed is None
    """

    def __init__(self):
        # Basic lists to track messages and seeds
        self.messages: List[bytes] = []            # Plaintext messages
        self.perfect_seeds: List[int] = []         # Matching seeds
        # For each message => (iv, ciphertext, entropy, coeffs)
        self.encryption_data: List[Tuple[bytes, bytes, float, Optional[List[float]]]] = []

        # Additional components
        self.timeline = TimelineManager()
        self.entropy_history: List[float] = []
        self.math_core = MathematicalCore()
        self.used_seeds = set()

        self.state: Dict[str, Any] = {}
        self.config_manager = ConfigManager()

        # Attempt to load state from file
        self.load_state()

    # ----------------------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------------------
    def save_state(self):
        """
        Save encryption state to 'quantum_stack_state.enc' (pickle).
        """
        logger.debug("Saving encryption state...")
        state_data = {
            'messages': [m.hex() for m in self.messages],
            'perfect_seeds': list(self.perfect_seeds),
            'encryption_data': [],
            'timeline_data': self.timeline.get_visualization_data(),
            'entropy_history': self.entropy_history,
            'used_seeds': list(self.used_seeds),
            'compressed_data': self.state.get('compressed_data', {
                'encrypted_storage': {},
                'pattern_storage': {},
                'active_identifiers': set()
            })
        }

        # Convert encryption_data to a serializable format
        for (iv, ct, ent, coeffs) in self.encryption_data:
            iv_hex = iv.hex()
            ct_hex = ct.hex()
            cf_list = list(coeffs) if coeffs is not None else None
            state_data['encryption_data'].append((iv_hex, ct_hex, ent, cf_list))
            logger.debug(f"Serialized encryption data: IV={iv_hex}, CT={ct_hex}, Entropy={ent}, Coeffs={cf_list}")

        # Convert sets to lists and bytes->hex in 'compressed_data'
        comp_data = state_data['compressed_data']
        if 'active_identifiers' in comp_data:
            comp_data['active_identifiers'] = list(comp_data['active_identifiers'])
        if 'encrypted_storage' in comp_data:
            for identifier, data in comp_data['encrypted_storage'].items():
                if 'combined_data' in data and isinstance(data['combined_data'], bytes):
                    data['combined_data'] = data['combined_data'].hex()
                    logger.debug(f"Serialized encrypted_storage for identifier {identifier}: {data}")

        # Write to file
        try:
            state_file = Path("quantum_stack_state.enc")
            with open(state_file, 'wb') as f:
                pickle.dump(state_data, f)
            logger.info("Encryption state saved successfully.")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def load_state(self):
        """
        Load encryption state from 'quantum_stack_state.enc' if it exists.
        """
        logger.debug("Loading encryption state...")
        try:
            state_file = Path("quantum_stack_state.enc")
            if not state_file.exists():
                logger.info("No existing state file found. Starting fresh.")
                return

            with open(state_file, 'rb') as f:
                state_data = pickle.load(f)

            # Rebuild messages
            self.messages = [bytes.fromhex(m_hex) for m_hex in state_data.get('messages', [])]
            logger.debug(f"Loaded messages: {self.messages}")

            self.perfect_seeds = state_data.get('perfect_seeds', [])
            self.entropy_history = state_data.get('entropy_history', [])
            used_seed_list = state_data.get('used_seeds', [])
            self.used_seeds = set(used_seed_list)
            logger.debug(f"Loaded perfect seeds: {self.perfect_seeds}")
            logger.debug(f"Loaded entropy history: {self.entropy_history}")
            logger.debug(f"Loaded used seeds: {self.used_seeds}")

            # Rebuild encryption_data
            self.encryption_data = []
            for (iv_hex, ct_hex, ent, cf_list) in state_data.get('encryption_data', []):
                iv = bytes.fromhex(iv_hex)
                ct = bytes.fromhex(ct_hex)
                cfs = [float(x) for x in cf_list] if cf_list is not None else None
                self.encryption_data.append((iv, ct, ent, cfs))
                logger.debug(f"Loaded encryption data: IV={iv_hex}, CT={ct_hex}, Entropy={ent}, Coeffs={cfs}")

            # Rebuild timeline
            if 'timeline_data' in state_data:
                self.timeline.restore_from_data(state_data['timeline_data'])
                logger.debug("Timeline data restored.")

            # Rebuild compressed_data
            comp_data = state_data.get('compressed_data', {
                'encrypted_storage': {},
                'pattern_storage': {},
                'active_identifiers': set()
            })

            if 'active_identifiers' in comp_data:
                comp_data['active_identifiers'] = set(comp_data['active_identifiers'])

            if 'encrypted_storage' in comp_data:
                for identifier, data in comp_data['encrypted_storage'].items():
                    if 'combined_data' in data and isinstance(data['combined_data'], str):
                        data['combined_data'] = bytes.fromhex(data['combined_data'])
                        logger.debug(f"Deserialized encrypted_storage for identifier {identifier}: {data}")

            self.state['compressed_data'] = comp_data
            logger.info("Encryption state loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            self.state = {}

    # ----------------------------------------------------------------------
    # Basic AES generation
    # ----------------------------------------------------------------------
    def generate_adaptive_key(self, seed: int, message_length: int) -> bytes:
        """
        Create a 32-byte AES key from seed + message_length.
        """
        rng = default_rng(seed)
        key = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        logger.debug(f"Generated adaptive key for seed {seed} and message length {message_length}: {key.hex()}")
        return key

    # ----------------------------------------------------------------------
    # Basic encrypt/decrypt with or without coefficients
    # ----------------------------------------------------------------------
    def encrypt_with_seed(self, message: bytes, seed: int) -> Tuple[bytes, bytes]:
        """
        Standard AES-CBC encryption with no c_k usage.
        Returns (iv, ciphertext).
        """
        logger.debug(f"Encrypting message with seed {seed} without coefficients.")
        key = self.generate_adaptive_key(seed, len(message))
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        ciphertext = cipher.encrypt(pad(message, AES.block_size))
        logger.debug(f"Encrypted message: IV={iv.hex()}, CT={ciphertext.hex()}")
        return iv, ciphertext

    def decrypt_with_seed(self, ciphertext: bytes, seed: int, iv: bytes) -> bytes:
        """Basic decryption without coefficients"""
        try:
            key = self.generate_adaptive_key(seed, len(ciphertext))
            cipher = AES.new(key, AES.MODE_CBC, iv)
            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
            logger.debug(f"Decryption successful for seed {seed}")
            return plaintext
        except Exception as e:
            logger.error(f"Decrypt with seed failed: {str(e)}")
            raise

    def encrypt_with_seed_and_coeffs(self, message: bytes, seed: int, coeffs: List[float]) -> Tuple[bytes, bytes]:
        """
        Advanced AES encryption => finalize base_key with c_k => AES-CBC.
        Returns (iv, ciphertext).
        """
        logger.debug(f"Encrypting message with seed {seed} using coefficients.")
        base_key = self.generate_adaptive_key(seed, len(message))
        final_key = combine_key_with_coeffs(base_key, coeffs)
        cipher = AES.new(final_key, AES.MODE_CBC)
        iv = cipher.iv
        ciphertext = cipher.encrypt(pad(message, AES.block_size))
        logger.debug(f"Encrypted message with coefficients: IV={iv.hex()}, CT={ciphertext.hex()}")
        return iv, ciphertext

    def decrypt_with_seed_and_coeffs(self, ciphertext: bytes, seed: int, iv: bytes, coeffs: List[float]) -> bytes:
        """Decrypt using seed and coefficients"""
        try:
            base_key = self.generate_adaptive_key(seed, len(ciphertext))
            final_key = combine_key_with_coeffs(base_key, coeffs)
            cipher = AES.new(final_key, AES.MODE_CBC, iv)
            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
            logger.debug(f"Decryption with coefficients successful for seed {seed}")
            return plaintext
        except Exception as e:
            logger.error(f"Decrypt with coefficients failed: {str(e)}")
            raise

    # ----------------------------------------------------------------------
    # "create_encrypted_segment(...)" for aggregator usage
    # ----------------------------------------------------------------------
    def create_encrypted_segment(self, seed: int, iv: bytes, ciphertext: bytes, entropy: float) -> bytes:
        """
        Create an encrypted segment with format:
            - seed (8 bytes)
            - entropy (8 bytes)
            - iv length (4 bytes) + iv
            - coefficients length (4 bytes) + coefficients data
            - ciphertext length (4 bytes) + ciphertext
        """
        logger.debug(f"Creating encrypted segment for seed {seed}")
        segment = bytearray()

        # Find the coefficients for this exact seed
        coeffs = None
        for i, (stored_seed, entry_data) in enumerate(zip(self.perfect_seeds, self.encryption_data)):
            if stored_seed == seed:
                coeffs = entry_data[3]  # coeffs is the 4th element
                break

        # Pack the data
        segment.extend(struct.pack('>Q', seed))  # 8 bytes seed
        segment.extend(struct.pack('>d', entropy))  # 8 bytes entropy

        # IV
        segment.extend(struct.pack('>I', len(iv)))
        segment.extend(iv)

        # Coefficients (always include them)
        if coeffs:
            coeffs_data = b''.join(struct.pack('>d', c) for c in coeffs)
            segment.extend(struct.pack('>I', len(coeffs_data)))
            segment.extend(coeffs_data)
            logger.debug(f"Added coefficients to segment for seed {seed}")
        else:
            segment.extend(struct.pack('>I', 0))  # No coefficients
            logger.warning(f"No coefficients found for seed {seed}")

        # Ciphertext
        segment.extend(struct.pack('>I', len(ciphertext)))
        segment.extend(ciphertext)

        return bytes(segment)

    # ----------------------------------------------------------------------
    # Perfect Entropy Seed Searching
    # ----------------------------------------------------------------------
    def find_perfect_entropy_seed(
        self,
        message: bytes,
        max_attempts: int = 50000
    ) -> Tuple[Optional[int], Optional[bytes], Optional[bytes], Optional[float]]:
        """
        Attempt to find a seed that yields ~1.0 encryption entropy for 'message',
        searching [1..2^32). Return (seed, iv, ciphertext, entropy) or (None, None, None, None).
        """
        logger.debug("Searching for a perfect entropy seed...")
        for attempt in range(1, max_attempts + 1):
            candidate = random.randint(1, (2**32) - 1)
            if candidate in self.used_seeds:
                continue

            iv, ct = self.encrypt_with_seed(message, candidate)
            bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8))
            ent = self.calculate_entropy(bits)
            logger.debug(f"Attempt {attempt}: Seed={candidate}, Entropy={ent}")

            if abs(ent - 1.0) < 1e-9:
                logger.info(f"Found perfect seed: {candidate} with entropy={ent:.10f}")
                self.used_seeds.add(candidate)
                return (candidate, iv, ct, ent)

        logger.warning("Perfect entropy seed not found within max_attempts.")
        return (None, None, None, None)
        
        
    def process_message(self, message: bytes, seed: Optional[int] = None, keyword: str = "") -> Tuple[bool, float]:
        """Process a single message with improved entropy handling"""
        
        if seed is None:
            # Find perfect entropy seed
            seed_result = self.find_perfect_entropy_seed(message)
            if seed_result[0] is None:
                return False, 0.0
                
            seed, iv, ct, entropy = seed_result
            logger.info(
                f"Generated perfect entropy seed: {seed}\n"
                f"Achieved entropy: {entropy:.6f}\n"
            )
        else:
            # Use provided seed but still calculate entropy
            iv, ct = self.encrypt_with_seed(message, seed)
            bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8))
            entropy = self.calculate_entropy(bits)
            logger.info(
                f"Using custom seed: {seed}\n"
                f"Achieved entropy: {entropy:.6f}\n"
            )
        
        # Store the message
        msg_id = len(self.messages)
        coeffs = generate_coefficients(100)  # Generate coefficients
        self.store_message(message, seed, iv, ct, entropy, msg_id, coeffs)
        
        return True, entropy

    def format_encryption_details(self, recipient: str, seed: int, keyword: str, entropy: float) -> str:
        """Format encryption details in a clean, professional manner"""
        return f"""
    Message Details for {recipient}:
    ----------------------------------------
    Seed:           {seed}
    Keyword:        {keyword}
    Final Entropy:  {entropy:.6f}
    ----------------------------------------
    """        

    # ----------------------------------------------------------------------
    # add_message => uses optional c_k or not
    # ----------------------------------------------------------------------
    def add_message(self, message: bytes, seed: Optional[int] = None, keyword: Optional[str] = None) -> Tuple[bool, float]:
        """
        Add a new message. Two paths:
        1. Custom seed: Use as-is with coefficients
        2. Random seed: Find perfect entropy seed without coeffs first, then apply coeffs
        """
        if not message:
            logger.error("Error: message is empty.")
            return (False, 0.0)

        max_len = 1048576
        if hasattr(self, 'config_manager'):
            max_len = self.config_manager.get_value("max_message_length", max_len)
        if len(message) > max_len:
            logger.error("Error: message too large.")
            return (False, 0.0)

        msg_id = len(self.messages)
        coeffs = generate_coefficients(100)  # Always generate coefficients

        if seed is not None:
            # Custom seed: use directly with coefficients
            iv, ct = self.encrypt_with_seed_and_coeffs(message, seed, coeffs)
            bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8))
            entropy = self.calculate_entropy(bits)
            logger.debug(f"Using custom seed {seed} with entropy {entropy}")
            self.store_message(message, seed, iv, ct, entropy, msg_id, coeffs)
            return (True, entropy)
        else:
            # Find perfect seed WITHOUT coefficients first
            s_res = self.find_perfect_entropy_seed(message)
            if s_res[0] is None:
                return (False, 0.0)

            perfect_seed = s_res[0]
            logger.debug(f"Found perfect seed {perfect_seed}, now applying coefficients")
            
            # Always apply coefficients to the perfect seed
            iv, ct = self.encrypt_with_seed_and_coeffs(message, perfect_seed, coeffs)
            bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8))
            final_entropy = self.calculate_entropy(bits)
            
            self.store_message(message, perfect_seed, iv, ct, final_entropy, msg_id, coeffs)
            return (True, final_entropy)

    def store_message(
        self,
        message: bytes,
        seed: int,
        iv: bytes,
        ciphertext: bytes,
        entropy: float,
        message_id: int,
        coeffs: Optional[List[float]] = None
    ):
        """
        Actually store the new message data => 
         - messages
         - perfect_seeds
         - encryption_data => (iv, ciphertext, ent, coeffs or None)
         - timeline marker
         - track entropies
        """
        logger.debug(f"Storing message ID {message_id}: Seed={seed}, Entropy={entropy}, Coeffs={'Yes' if coeffs else 'No'}")
        self.messages.append(message)
        self.perfect_seeds.append(seed)
        self.encryption_data.append((iv, ciphertext, entropy, coeffs))
        self.timeline.create_marker(seed, message_id, message, entropy)
        self.entropy_history.append(entropy)
        logger.info(f"Message ID {message_id} stored successfully.")

    # ----------------------------------------------------------------------
    # aggregator => combine & parse
    # ----------------------------------------------------------------------
    def combine_messages(self) -> bytes:
        """
        Create aggregator data, including coefficients:
        For each message => create encrypted segment
        """
        logger.debug("Combining all messages into aggregator data.")
        combined = bytearray()
        for i in range(len(self.messages)):
            seed = self.perfect_seeds[i]
            iv, ciphertext, entropy, _ = self.encryption_data[i]

            # Create encrypted segment
            segment = self.create_encrypted_segment(seed, iv, ciphertext, entropy)
            combined.extend(segment)
            logger.debug(f"Added encrypted segment for message ID {i}: Seed={seed}")

        logger.info("All messages combined into aggregator data.")
        return bytes(combined)


    def extract_message(self, combined_data: bytes, seed: int) -> Tuple[Optional[bytes], Optional[int]]:
        """Extract and decrypt message with proper type handling"""
        try:
            if isinstance(combined_data, str):
                combined_data = bytes.fromhex(combined_data)

            pos = 0
            data_len = len(combined_data)

            while pos + 24 <= data_len:  # Minimum length check for header
                # Read seed and entropy
                msg_seed = struct.unpack('>Q', combined_data[pos:pos+8])[0]
                pos += 8
                entropy = struct.unpack('>d', combined_data[pos:pos+8])[0]
                pos += 8

                # Read IV with proper length check
                if pos + 4 > data_len:
                    break
                iv_len = struct.unpack('>I', combined_data[pos:pos+4])[0]
                pos += 4
                if pos + iv_len > data_len:
                    break
                iv = combined_data[pos:pos+iv_len]
                pos += iv_len

                # Read coefficients with proper length check
                if pos + 4 > data_len:
                    break
                coeffs_len = struct.unpack('>I', combined_data[pos:pos+4])[0]
                pos += 4
                
                coeffs = []
                if coeffs_len > 0:
                    if pos + coeffs_len > data_len:
                        break
                    coeffs_data = combined_data[pos:pos+coeffs_len]
                    pos += coeffs_len
                    num_coeffs = coeffs_len // 8  # 8 bytes per double
                    coeffs = [struct.unpack('>d', coeffs_data[i*8:(i+1)*8])[0] 
                             for i in range(num_coeffs)]

                # Read ciphertext with proper length check
                if pos + 4 > data_len:
                    break
                ct_len = struct.unpack('>I', combined_data[pos:pos+4])[0]
                pos += 4
                if pos + ct_len > data_len:
                    break
                ciphertext = combined_data[pos:pos+ct_len]
                pos += ct_len

                if msg_seed == seed:
                    if coeffs:
                        try:
                            plaintext = self.decrypt_with_seed_and_coeffs(
                                ciphertext, seed, iv, coeffs
                            )
                            return plaintext, None
                        except Exception as e:
                            logger.error(f"Decryption failed: {e}")
                            return None, None
                    else:
                        try:
                            plaintext = self.decrypt_with_seed(ciphertext, seed, iv)
                            return plaintext, None
                        except Exception as e:
                            logger.error(f"Decryption failed: {e}")
                            return None, None

            return None, None

        except Exception as e:
            logger.error(f"Extract message error: {e}")
            return None, None

    def format_hash(self, combined_data: bytes) -> str:
        """Convert aggregator bytes => hex."""
        logger.debug("Formatting aggregator data to hex.")
        return combined_data.hex()

    def verify_hash(self, hash_data: str) -> bool:
        """Rebuild aggregator => compare with given hash hex."""
        logger.debug("Verifying aggregator hash.")
        try:
            re_combo = self.combine_messages()
            re_hex = self.format_hash(re_combo)
            is_valid = re_hex == hash_data
            logger.info(f"Aggregator hash verification result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Error verifying aggregator hash: {e}")
            return False

    # ----------------------------------------------------------------------
    # Additional statistical tests
    # ----------------------------------------------------------------------
    def calculate_entropy(self, bits: np.ndarray) -> float:
        """Shannon entropy on a binary distribution => bits per bit in [0..1]."""
        unique, counts = np.unique(bits, return_counts=True)
        p = counts / len(bits)
        ent = -np.sum(p * np.log2(p))
        logger.debug(f"Calculated Shannon entropy: {ent}")
        return ent

    def monobit_test(self, data: np.ndarray) -> float:
        """Monobit test => measure 0/1 imbalance => returns p-value."""
        ones = np.count_nonzero(data)
        zeros = len(data) - ones
        s = abs(ones - zeros) / np.sqrt(len(data))
        p_value = erfc(s / np.sqrt(2))
        logger.debug(f"Monobit test: Ones={ones}, Zeros={zeros}, p-value={p_value}")
        return p_value

    def runs_test(self, data: np.ndarray) -> float:
        """Runs test => detect flips in consecutive bits => returns p-value."""
        ones = np.count_nonzero(data)
        zeros = len(data) - ones
        pi = ones / len(data)

        if abs(pi - 0.5) >= (2 / np.sqrt(len(data))):
            logger.debug("Runs test failed due to pi threshold.")
            return 0.0

        vobs = 1
        for i in range(1, len(data)):
            if data[i] != data[i - 1]:
                vobs += 1

        num = abs(vobs - (2 * len(data) * pi * (1 - pi)))
        den = 2 * np.sqrt(2 * len(data)) * pi * (1 - pi)
        p_value = erfc(num / den)
        logger.debug(f"Runs test: Vobs={vobs}, p-value={p_value}")
        return p_value

    def chi_squared_test(self, data: np.ndarray) -> float:
        """Chi-squared test => df=1 for binary distribution."""
        ones = np.count_nonzero(data)
        zeros = len(data) - ones
        expected = len(data) / 2
        chi_sq = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected
        p_value = 1 - chi2.cdf(chi_sq, df=1)
        logger.debug(f"Chi-squared test: ChiSq={chi_sq}, p-value={p_value}")
        return p_value

    def avalanche_test(self, message: bytes, seed: int) -> float:
        """
        Avalanche test => flip one bit => measure difference in ciphertext bits.
        Returns the ratio of differing bits in [0..1].
        """
        logger.debug(f"Performing avalanche test for seed {seed}.")
        # Original ciphertext
        iv1, ct1 = self.encrypt_with_seed(message, seed)
        bits1 = np.unpackbits(np.frombuffer(ct1, dtype=np.uint8))

        # Flipped message
        flipped = bytearray(message)
        flipped[0] ^= 0x01
        iv2, ct2 = self.encrypt_with_seed(bytes(flipped), seed)
        bits2 = np.unpackbits(np.frombuffer(ct2, dtype=np.uint8))

        # Calculate differing bits
        diff = np.sum(bits1 != bits2)
        ratio = diff / len(bits1)
        logger.debug(f"Avalanche test: Differing bits={diff}, Ratio={ratio}")
        return ratio
