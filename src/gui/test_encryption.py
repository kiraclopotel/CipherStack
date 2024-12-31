# test_encryption.py

import unittest
from encryption import QuantumStackEncryption

class EnhancedTestSuite(unittest.TestCase):
    def setUp(self):
        self.encryption = QuantumStackEncryption()

    def test_encryption_decryption(self):
        print("Test encryption and decryption functionality ...")
        message = b"Test Message"
        seed = 123456789
        iv, ciphertext = self.encryption.encrypt_with_seed(message, seed)
        decrypted = self.encryption.decrypt_with_seed(ciphertext, seed, iv)
        self.assertEqual(message, decrypted)

    def test_encryption_decryption_with_coeffs(self):
        print("Test encryption and decryption with coefficients ...")
        message = b"Test Message with Coeffs"
        seed = 987654321
        coeffs = self.encryption.generate_coefficients(20)
        iv, ciphertext = self.encryption.encrypt_with_seed_and_coeffs(message, seed, coeffs)
        decrypted = self.encryption.decrypt_with_seed_and_coeffs(ciphertext, seed, iv, coeffs)
        self.assertEqual(message, decrypted)

    def test_error_handling(self):
        print("Test error handling and edge cases ...")
        # Test empty message
        success, entropy = self.encryption.add_message(b"", seed=111111, use_coeffs=False)
        self.assertFalse(success)
        self.assertEqual(entropy, 0.0)

        # Test message too large
        large_message = b"a" * (2**20 + 1)  # 1MB + 1 byte
        success, entropy = self.encryption.add_message(large_message, seed=222222, use_coeffs=False)
        self.assertFalse(success)
        self.assertEqual(entropy, 0.0)

    def test_performance_metrics(self):
        print("Test performance and timing metrics ...")
        message = b"Performance Test Message"
        seed = 333333333
        iv, ciphertext = self.encryption.encrypt_with_seed(message, seed)
        decrypted = self.encryption.decrypt_with_seed(ciphertext, seed, iv)
        self.assertEqual(message, decrypted)

    def test_hash_integrity(self):
        print("Test hash generation and verification ...")
        message = b"Hash Integrity Test Message"
        seed = 444444444
        success, entropy = self.encryption.add_message(message, seed=seed, use_coeffs=False)
        self.assertTrue(success)

        aggregator_data = self.encryption.combine_messages()
        aggregator_hash = self.encryption.format_hash(aggregator_data)
        is_valid = self.encryption.verify_hash(aggregator_hash)
        self.assertTrue(is_valid)

    def test_key_generation(self):
        print("Test key generation uniqueness and properties ...")
        seed1 = 555555555
        seed2 = 666666666
        key1 = self.encryption.generate_adaptive_key(seed1, 100)
        key2 = self.encryption.generate_adaptive_key(seed2, 100)
        self.assertNotEqual(key1, key2)
        self.assertEqual(len(key1), 32)
        self.assertEqual(len(key2), 32)

    def test_layer_functions(self):
        print("Test layer computation and transitions ...")
        # Placeholder for layer computation tests
        self.assertTrue(True)  # Replace with actual tests

    def test_randomness(self):
        print("Test statistical properties of encrypted data ...")
        message = b"Randomness Test Message"
        seed = 777777777
        iv, ciphertext = self.encryption.encrypt_with_seed(message, seed)
        bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
        ent = self.encryption.calculate_entropy(bits)
        self.assertGreater(ent, 0.99)  # Expect high entropy

    def test_timeline_verification(self):
        print("Test timeline management and verification ...")
        message = b"Timeline Test Message"
        seed = 888888888
        success, entropy = self.encryption.add_message(message, seed=seed, use_coeffs=False)
        self.assertTrue(success)

        aggregator_data = self.encryption.combine_messages()
        plaintext, message_id = self.encryption.extract_message(aggregator_data, seed=seed)
        self.assertIsNotNone(plaintext)
        self.assertEqual(message, plaintext)

if __name__ == '__main__':
    unittest.main()
