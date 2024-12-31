# src/core/mathematics.py

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import math
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
import time

@dataclass
class LayerMetrics:
    """
    Stores metrics for layer function calculations.
    These metrics help monitor the health and security
    of the encryption system.
    """
    entropy: float           # Measure of randomness in the system
    complexity: float        # Measure of computational complexity
    stability: float         # Measure of numerical stability
    convergence_rate: float  # Rate at which calculations converge


class MathematicalCore:
    """
    Enhanced implementation of the mathematical core functionality
    for quantum stack encryption. Combines theoretical foundations
    from the thesis with additional security features and maintains
    backward compatibility with existing functionality.
    """

    def __init__(self, max_layer_depth: int = 100):
        # Initialize caches and state management
        self.layer_cache: Dict[Tuple[float, int], float] = {}
        self.timeline_cache: Dict[int, List[Tuple[int, int]]] = {}
        self.layer_states: Dict[int, int] = {}
        self.layer_transitions: List[Tuple[int, int, int]] = []

        # Initialize enhanced grid system for cryptographic patterns
        self.grid_dimensions = (10, 10, 10)
        self.grid_values = np.zeros(self.grid_dimensions)
        self.initialize_enhanced_grid()

        # System configuration parameters
        self.max_layer_depth = max_layer_depth
        self.entropy_threshold = 0.9999
        self.numerical_tolerance = 1e-10

        # Mathematical stability tracking
        self.previous_layer: Optional[int] = None
        self.stability_metrics: List[LayerMetrics] = []

    def initialize_enhanced_grid(self) -> None:
        """
        Initialize the 3D grid with advanced mathematical patterns
        that increase cryptographic strength. Combines linear, trigonometric,
        and exponential components for maximum complexity.
        """
        for i in range(self.grid_dimensions[0]):
            for j in range(self.grid_dimensions[1]):
                for k in range(self.grid_dimensions[2]):
                    # Create complex patterns using multiple math components
                    linear_component = (j - i) % 10
                    trig_component = int(5 * (math.sin(i / 5) + math.cos(j / 5)))
                    expo_component = int(math.exp(k / 10)) % 10

                    # Mix them via modular arithmetic for non-linearity
                    value = (linear_component + trig_component + expo_component) % 10
                    self.grid_values[i, j, k] = value

    def generate_entry_timeline(self, n: int) -> List[Tuple[int, int]]:
        """
        Generate timeline entries for a given integer 'n',
        as specified in the system's theoretical foundation.
        Returns a list of (digit, depth) pairs.
        """
        # Check cache first
        if n in self.timeline_cache:
            return self.timeline_cache[n]

        # Build timeline from digits
        str_n = str(n)
        k = len(str_n)
        timeline = [(int(str_n[i]), k - i) for i in range(k)]

        self.timeline_cache[n] = timeline
        return timeline

    def compute_layer(self, value: int, layer: int = 1) -> int:
        """
        Compute the layer based on the number of digits in 'value'.
        Example:
          - 1..9 => layer=1
          - 10..99 => layer=2
          - etc.
        """
        num_digits = len(str(abs(value)))  # handle negative safely
        self.previous_layer = num_digits
        return num_digits

    def layer_function(self, n: float, k: int, include_enhancements: bool = True) -> float:
        """
        The 'layer function' that merges theoretical requirements
        with additional security features. 
         - n must be > 0
         - k must be > 0
         - Returns a float, possibly large.

        If 'include_enhancements' is True, we add extra security
        transformations (b_k, c_k, stability term, etc.).
        """
        if n <= 0:
            warnings.warn("Input value must be positive. Setting n=1.000001")
            n = 1.000001

        if k < 1:
            warnings.warn("Layer index must be positive. Setting k=1.")
            k = 1
        if k > self.max_layer_depth:
            warnings.warn(f"Layer {k} exceeds max depth {self.max_layer_depth}. Using max.")
            k = self.max_layer_depth

        cache_key = (n, k)
        if cache_key in self.layer_cache:
            return self.layer_cache[cache_key]

        try:
            # prevent huge overflow
            n = min(n, 1e308)

            # basic part
            log_n = math.log(n)
            power_term = n ** (1 / k)
            log_term = (log_n) ** ((k - 1) / k)

            result = power_term * log_term

            if include_enhancements:
                b_k = self._compute_adaptive_coefficient(k)
                c_k = self._compute_layer_coefficient(k, n)
                enhancement_term = b_k * math.log(log_n + 1)
                complexity_term = 0.01 * (log_n ** 2) * (n ** (1 / k))
                stability_term = self._compute_stability_term(k, n)

                result = c_k * result + enhancement_term + complexity_term + stability_term

            self.layer_cache[cache_key] = result
            self._update_stability_metrics(n, k, result)

            return result

        except Exception as e:
            warnings.warn(f"Error in layer_function: {str(e)}")
            return 0.0

    def logarithmic_ratio(self, n: float, k: int) -> float:
        """
        Compute the ratio f_k(n) / f_(k+1)(n) from the theorem,
        or approximate it if n>1.
        """
        if n <= 1 or k < 1:
            return 0.0

        try:
            f_k = self.layer_function(n, k)
            f_kp1 = self.layer_function(n, k + 1)
            if f_kp1 == 0:
                return float('inf')

            # per the formula: (n / log(n))^(1/(k(k+1)))
            ratio = (n / math.log(n)) ** (1 / (k * (k + 1)))
            if ratio < float('inf') and ratio > 0:
                return ratio
            return 0.0
        except (ValueError, OverflowError):
            return 0.0

    def wave_function_hybrid(self, x: float, k: int, omega: float) -> float:
        """
        Another advanced transform: wave + power/log combination
        for added complexity.
        """
        if x <= 0:
            return 0.0

        x = min(x, 1e300)
        k = min(k, 1000)
        omega = min(omega, 1000)

        power_term = x ** (1 / k)
        log_term = (math.log(x)) ** ((k - 1) / k)
        wave_term = math.sin(omega * x)

        return power_term * log_term * wave_term

    # -----------------------
    # HELPER Coeff Calculations
    # -----------------------

    def _compute_adaptive_coefficient(self, k: int) -> float:
        """
        b_k: depends on layer depth k and stability metrics
        """
        b_k = 0.1 * k
        layer_factor = math.exp(-k / 10)
        b_k *= layer_factor

        if self.stability_metrics:
            latest = self.stability_metrics[-1]
            stability_factor = 1 + (1 - latest.stability) * 0.1
            b_k *= stability_factor

        return min(max(b_k, 0.01), 1.0)

    def _compute_layer_coefficient(self, k: int, n: float) -> float:
        """
        c_k: depends on n & k. Additional dynamic scaling
        """
        c_k = 1.0
        size_factor = math.log(n + 1) / (k * k)
        c_k *= (1 + size_factor)

        depth_factor = 1 - math.exp(-k / 5)
        c_k *= depth_factor

        if self.stability_metrics:
            latest = self.stability_metrics[-1]
            conv_factor = 1 + (1 - latest.convergence_rate) * 0.1
            c_k *= conv_factor

        return min(max(c_k, 0.5), 2.0)

    def _compute_stability_term(self, k: int, n: float) -> float:
        """Add small term based on last stability metrics."""
        if not self.stability_metrics:
            return 0.0

        latest = self.stability_metrics[-1]
        stability_factor = latest.entropy * latest.stability
        oscillation = math.sin(k * math.pi / 10) * math.cos(math.log(n))

        return 0.01 * stability_factor * oscillation

    def _update_stability_metrics(self, n: float, k: int, result: float) -> None:
        """
        Update the system's stability metrics each time layer_function is called.
        This helps track the 'health' or 'stability' of the system.
        """
        if not self.stability_metrics:
            # Initialize
            entropy = 1.0
            stability = 1.0
            convergence_rate = 1.0
        else:
            previous = self.stability_metrics[-1]
            entropy = min(1.0, previous.entropy * 0.9 + 0.1)
            stability = min(1.0, abs(math.sin(result)))
            convergence_rate = min(1.0, 1.0 / k)

        complexity = min(1.0, math.log(n) / (k * 100))

        metrics = LayerMetrics(
            entropy=entropy,
            complexity=complexity,
            stability=stability,
            convergence_rate=convergence_rate
        )
        self.stability_metrics.append(metrics)

        # keep size manageable
        if len(self.stability_metrics) > 1000:
            self.stability_metrics = self.stability_metrics[-1000:]

    # -----------------------
    # LAYER STATE
    # -----------------------

    def _smooth_layer_transition(self, old_layer: int, new_layer: int) -> int:
        """
        If difference is >1, move gradually by 1 step at a time.
        """
        if abs(new_layer - old_layer) > 1:
            return old_layer + (1 if new_layer > old_layer else -1)
        return new_layer

    def _manage_layer_state(self, seed: int, new_layer: int) -> int:
        """
        Manage transitions between layer states for a given seed.
        """
        if seed in self.layer_states:
            old_layer = self.layer_states[seed]
            transition_layer = self._smooth_layer_transition(old_layer, new_layer)
            self.layer_states[seed] = transition_layer
            self.layer_transitions.append((seed, old_layer, transition_layer))
            return transition_layer

        self.layer_states[seed] = new_layer
        return new_layer

    def clear_caches(self) -> None:
        """Clear all caches and states, resetting the system."""
        self.layer_cache.clear()
        self.timeline_cache.clear()
        self.layer_states.clear()
        self.layer_transitions.clear()
        self.stability_metrics.clear()
        self.previous_layer = None

    def get_stability_analysis(self) -> Dict[str, float]:
        """
        Provide a summary of the stability metrics, e.g. average entropy
        and complexity over the last 100 calls.
        """
        if not self.stability_metrics:
            return {
                'average_entropy': 1.0,
                'average_complexity': 0.0,
                'average_stability': 1.0,
                'average_convergence': 1.0,
                'system_health': 1.0
            }

        metrics = self.stability_metrics[-100:]
        avg_entropy = np.mean([m.entropy for m in metrics])
        avg_complexity = np.mean([m.complexity for m in metrics])
        avg_stability = np.mean([m.stability for m in metrics])
        avg_convergence = np.mean([m.convergence_rate for m in metrics])
        system_health = np.mean([m.entropy * m.stability * m.convergence_rate for m in metrics])

        return {
            'average_entropy': avg_entropy,
            'average_complexity': avg_complexity,
            'average_stability': avg_stability,
            'average_convergence': avg_convergence,
            'system_health': system_health
        }

    # -----------------------
    # MESSAGE PRE-PROCESSING
    # -----------------------

    def process_message_for_encryption(self, message: bytes) -> Tuple[bytes, int, float]:
        """
        Process a message for encryption, ensuring independence and near-perfect entropy
        if possible. This is the main entry for advanced message processing prior
        to actual encryption in the main system.
        """
        # Step 1: Balance bits
        processed_data = self.balance_bit_distribution(message)

        # Step 2: Possibly find a perfect or near-perfect seed
        perfect_seed, initial_entropy = self.find_independent_perfect_seed(processed_data)

        # Step 3: record
        self.record_message_metrics(perfect_seed, initial_entropy, len(processed_data))

        # Step 4: check independence
        if not self.verify_message_independence(perfect_seed):
            perfect_seed = self.adjust_seed_for_independence(perfect_seed)
            processed_data = self.apply_independence_enhancement(processed_data)

        # final entropy
        final_entropy = self.calculate_message_entropy(processed_data)

        return processed_data, perfect_seed, final_entropy

    def record_message_metrics(self, seed: int, entropy: float, size: int) -> None:
        """Record metrics for each message to track independence and patterns."""
        if not hasattr(self, 'message_metrics'):
            self.message_metrics = []

        metrics = {
            'seed': seed,
            'entropy': entropy,
            'size': size,
            'timestamp': time.time(),
            'hash': hashlib.sha256(str(seed).encode()).hexdigest()[:16]
        }
        self.message_metrics.append(metrics)

        if len(self.message_metrics) > 1000:
            self.message_metrics = self.message_metrics[-1000:]

    def verify_message_independence(self, new_seed: int) -> bool:
        """Check if new seed is independent from recent seeds."""
        if not hasattr(self, 'message_metrics') or not self.message_metrics:
            return True

        recent_metrics = self.message_metrics[-10:]
        for m in recent_metrics:
            if abs(new_seed - m['seed']) < 1000:
                return False
            new_hash = hashlib.sha256(str(new_seed).encode()).hexdigest()[:16]
            if self.calculate_hash_similarity(new_hash, m['hash']) > 0.3:
                return False
        return True

    def calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity of two short hex hashes by Hamming distance."""
        if len(hash1) != len(hash2):
            return 0.0
        diffs = sum(a != b for a, b in zip(hash1, hash2))
        return 1 - (diffs / len(hash1))

    def adjust_seed_for_independence(self, seed: int) -> int:
        """Incrementally adjust seed to find a more independent value."""
        adjustment = 1000
        new_seed = seed
        while not self.verify_message_independence(new_seed):
            new_seed = seed + adjustment
            adjustment *= 2

            if adjustment > 1_000_000:
                # fallback
                new_seed = seed + (int(time.time() * 1000) % 1_000_000)
                break

        return new_seed

    def apply_independence_enhancement(self, data: bytes) -> bytes:
        """
        Additional transformations to enhance independence,
        possibly toggling bits in a non-destructive way.
        """
        try:
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            shifted_bits = np.roll(bits, len(bits) // 3)
            xor_pattern = np.array([int(math.sin(i / 10) > 0) for i in range(len(bits))])
            enhanced_bits = np.logical_xor(shifted_bits, xor_pattern).astype(np.uint8)
            return np.packbits(enhanced_bits).tobytes()
        except Exception as e:
            warnings.warn(f"Error applying independence enhancement: {str(e)}")
            return data

    def balance_bit_distribution(self, data: bytes) -> bytes:
        """
        Balance distribution of bits by toggling half the bits if imbalance is large.
        """
        try:
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            ones = np.int64(np.count_nonzero(bits))
            total = np.int64(len(bits))
            zeros = total - ones
            threshold = np.int64(total * 0.1)

            if abs(ones - zeros) > threshold:
                pattern = np.zeros(total, dtype=np.uint8)
                pattern[::2] = 1
                balanced_bits = np.logical_xor(bits, pattern).astype(np.uint8)
                return np.packbits(balanced_bits).tobytes()
            return data
        except Exception as e:
            warnings.warn(f"Error in bit balancing: {str(e)}")
            return data

    def calculate_message_entropy(self, message_data: bytes) -> float:
        """
        Calculate a 0..(some scale) measure of entropy for 'message_data'.
        Typically 0..8 bits per byte for random data, but you can interpret as you want.
        """
        try:
            bits = np.unpackbits(np.frombuffer(message_data, dtype=np.uint8))
            unique, counts = np.unique(bits, return_counts=True)
            probabilities = counts / len(bits)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            # not strictly normalized, but you can do / math.log2(256) if you want per-byte measure
            return entropy
        except Exception as e:
            warnings.warn(f"Error calculating entropy: {str(e)}")
            return 0.0

    def find_independent_perfect_seed(self, data: bytes) -> Tuple[int, float]:
        """
        A placeholder function that 'finds' a perfect seed for data.
        Currently just returns a random seed + basic measure of 'entropy'.
        In a real system, you'd do a search for near-perfect encryption entropy, etc.
        """
        seed = random.randint(1, 2**64 - 1)
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        unique, counts = np.unique(bits, return_counts=True)
        p = counts / len(bits)
        e = -np.sum(p * np.log2(p))
        return seed, e
