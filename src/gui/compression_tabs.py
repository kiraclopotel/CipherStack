# compression_tabs.py

import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, filedialog, messagebox
from typing import Dict, Any, List, Tuple, Optional
import hashlib
import time
import random
import string
import os
import numpy as np

from utils.helpers import DataValidator, EncryptionHelper
from core.compress import EnhancedCompressor, CompressionResult
from core.quantum_stack_integration import QuantumStackIntegrator
from core.secure_storage.storage import create_secure_storage, FileHandler

class MessageRow:
    """
    Class to encapsulate all UI elements for a single message input row,
    including optional seed and keyword, plus an entropy label and delete button.
    """
    def __init__(self, parent: ttk.Frame, index: int, delete_callback):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", pady=2)

        # Message Label + Entry
        self.message_label = ttk.Label(self.frame, text=f"Message {index + 1}:")
        self.message_label.pack(side="left")
        self.message_entry = ttk.Entry(self.frame, width=40)
        self.message_entry.pack(side="left", padx=5)

        # Optional Seed
        self.seed_label = ttk.Label(self.frame, text="Seed (Optional):")
        self.seed_label.pack(side="left", padx=5)
        self.seed_entry = ttk.Entry(self.frame, width=10)
        self.seed_entry.pack(side="left", padx=5)

        # Optional Keyword
        self.keyword_label = ttk.Label(self.frame, text="Keyword (Optional):")
        self.keyword_label.pack(side="left", padx=5)
        self.keyword_entry = ttk.Entry(self.frame, width=15)
        self.keyword_entry.pack(side="left", padx=5)

        # Entropy Display
        self.entropy_label = ttk.Label(self.frame, text="Entropy: -")
        self.entropy_label.pack(side="left", padx=5)

        # Delete Button
        self.delete_button = ttk.Button(
            self.frame,
            text="×",
            width=2,
            command=lambda: delete_callback(self)
        )
        self.delete_button.pack(side="right", padx=5)

    def update_label(self, index: int):
        """Update the row label when rows are added or removed."""
        self.message_label.configure(text=f"Message {index + 1}:")


class CompressedEncryptionTab(ttk.Frame):
    """
    The main tab for compressing and encrypting messages.
    Collects multiple messages, seeds, and keywords.
    Compresses them, then encrypts them, producing a final "identifier."
    """

    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.validator = DataValidator()
        self.helper = EncryptionHelper()
        self.compressor = EnhancedCompressor()
        self.integrator = QuantumStackIntegrator()

        # UI elements for message rows
        self.message_rows: List[MessageRow] = []

        # Processing state
        self.processing = False
        self.last_identifier = ""

        # Retrieve encryption system from shared_state
        encryption_system = self.shared_state.get('encryption')
        if encryption_system and hasattr(encryption_system, 'state'):
            if 'compressed_data' not in encryption_system.state:
                encryption_system.state['compressed_data'] = {
                    'encrypted_storage': {},
                    'pattern_storage': {},
                    'active_identifiers': set()
                }

            # Load references from encryption state
            comp_data = encryption_system.state['compressed_data']
            self.shared_state['encrypted_storage'] = comp_data['encrypted_storage']
            self.shared_state['pattern_storage'] = comp_data['pattern_storage']
            self.shared_state['active_identifiers'] = comp_data['active_identifiers']

        self.setup_ui()

    def setup_ui(self):
        """
        Build all UI elements: message entry rows, compression settings,
        identifier controls, and results area.
        """

        # Frame for Messages
        input_frame = ttk.LabelFrame(self, text="Message Input", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        # Create initial rows
        for i in range(5):
            self.create_message_row(input_frame, i)

        # Buttons under message rows
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(buttons_frame, text="Add More Rows",
                   command=self.add_more_rows).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Fill Random",
                   command=self.fill_random_data).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Clear All",
                   command=self.clear_all).pack(side="left", padx=5)

        # Compression Settings
        compression_frame = ttk.LabelFrame(self, text="Compression Settings", padding=10)
        compression_frame.pack(fill="x", padx=10, pady=5)

        self.target_size_var = tk.StringVar(value="15")
        ttk.Label(compression_frame, text="Identifier Size:").pack(side="left")
        ttk.Entry(compression_frame, textvariable=self.target_size_var, width=10).pack(side="left", padx=5)

        # Identifier Controls
        identifier_frame = ttk.LabelFrame(self, text="Identifier Operations", padding=10)
        identifier_frame.pack(fill="x", padx=10, pady=5)

        self.identifier_text = scrolledtext.ScrolledText(identifier_frame, height=3, state='disabled')
        self.identifier_text.pack(fill="x", padx=5, pady=5)

        identifier_controls = ttk.Frame(identifier_frame)
        identifier_controls.pack(fill="x", padx=5, pady=5)

        ttk.Button(identifier_controls, text="Compress & Encrypt",
                   command=self.compress_and_encrypt).pack(side="left", padx=5)
        ttk.Button(identifier_controls, text="Copy Identifier",
                   command=self.copy_identifier).pack(side="left", padx=5)
        ttk.Button(identifier_controls, text="Save Identifier",
                   command=self.save_identifier).pack(side="left", padx=5)

        # Results / Status
        results_frame = ttk.LabelFrame(self, text="Results & Status", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, state='disabled')
        self.results_text.pack(fill="both", expand=True)

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(results_frame,
                                            variable=self.progress_var,
                                            maximum=100)
        self.progress_bar.pack(fill="x", padx=5, pady=5)

        # Status Label
        self.status_label = ttk.Label(results_frame, text="Ready")
        self.status_label.pack(pady=5)

    def create_message_row(self, parent: ttk.Frame, index: int):
        """Add a new MessageRow UI object."""
        row = MessageRow(parent, index, self.delete_row)
        self.message_rows.append(row)


    def compress_and_encrypt(self):
        """Main handler to compress and encrypt messages."""
        if self.processing:
            messagebox.showwarning("Processing", "Already processing. Please wait.")
            return

        self.processing = True
        self.clear_results()
        self.update_status("Processing messages...")

        try:
            target_size = int(self.target_size_var.get())
            if target_size <= 0:
                raise ValueError("Invalid target size")
        except ValueError:
            self.update_status("Invalid identifier size (must be positive int).")
            self.processing = False
            return

        messages = []
        recipient_data = {}
        total_rows = len([r for r in self.message_rows if r.message_entry.get().strip()])

        if total_rows == 0:
            self.update_status("No messages to process.")
            self.processing = False
            return

        encryption_system = self.shared_state.get('encryption')
        if not encryption_system:
            self.update_status("Encryption system not found.")
            self.processing = False
            return

        self.append_result("Starting message processing...\n")

        for i, row in enumerate(self.message_rows):
            message_text = row.message_entry.get().strip()
            if not message_text:
                continue

            seed_str = row.seed_entry.get().strip()
            keyword_str = row.keyword_entry.get().strip() or f'default_key_{i}'

            self.append_result(f"Processing message {i+1}:\n")
            self.append_result(f"Message text: {message_text}\n")
            self.append_result(f"Seed string: {seed_str}\n")

            try:
                if not seed_str:
                    self.append_result("No seed provided. Searching for perfect entropy seed...\n")
                    message_bytes = message_text.encode()
                    seed_result = encryption_system.find_perfect_entropy_seed(message_bytes)
                    if seed_result[0] is None:
                        self.append_result("Failed to find perfect entropy seed.\n")
                        continue
                        
                    seed_val = seed_result[0]
                    entropy = seed_result[3]
                    self.append_result(f"Found perfect entropy seed: {seed_val}\n")
                    self.append_result(f"Achieved entropy: {entropy:.6f}\n")
                    row.entropy_label.configure(text=f"Entropy: {entropy:.6f}")
                else:
                    seed_val = int(seed_str)
                    if seed_val <= 0:
                        raise ValueError("Seed must be positive")
                    message_bytes = message_text.encode()
                    
                    # Calculate entropy for custom seed
                    iv, ct = encryption_system.encrypt_with_seed(message_bytes, seed_val)
                    bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8))
                    entropy = encryption_system.calculate_entropy(bits)
                    self.append_result(f"Using custom seed: {seed_val}\n")
                    self.append_result(f"Achieved entropy: {entropy:.6f}\n")
                    row.entropy_label.configure(text=f"Entropy: {entropy:.6f}")

                messages.append(message_text.encode())
                recipient_data[f'recipient_{i}'] = (seed_val, keyword_str)

                progress_pct = (len(messages) / total_rows) * 100
                self.progress_var.set(progress_pct)
                self.update()

            except ValueError as e:
                self.append_result(f"Invalid seed format: {str(e)}. Using random seed.\n")
                seed_val = random.randint(1, 2**32 - 1)
                messages.append(message_text.encode())
                recipient_data[f'recipient_{i}'] = (seed_val, keyword_str)
                row.seed_entry.delete(0, tk.END)
                row.seed_entry.insert(0, str(seed_val))

        if not messages:
            self.update_status("No valid messages to process.")
            self.processing = False
            return

        self.append_result(f"\nProcessing {len(messages)} messages with compression...\n")

        identifier = self.integrator.encrypt_messages_for_recipients(
            messages,
            recipient_data,
            target_size
        )

        if identifier:
            self.identifier_text.configure(state='normal')
            self.identifier_text.delete("1.0", tk.END)
            self.identifier_text.insert(tk.END, identifier)
            self.identifier_text.configure(state='disabled')
            self.last_identifier = identifier

            self.append_result(
                f"\nIdentifier ({target_size} chars): {identifier}\n"
                "Messages encrypted and stored successfully.\n\n"
                "Decryption Information:\n"
                "======================\n"
            )

            for i, (recipient, (seed_val, keyword_str)) in enumerate(recipient_data.items()):
                self._display_seed_info(seed_val, keyword_str, i)

            self.update_status("Processing completed successfully")
        else:
            self.append_result("Error occurred during encryption process.\n")
            self.update_status("Processing failed")

        self.processing = False


    def _display_seed_info(self, seed_val: int, keyword: str, idx: int):
            """Display seed and keyword info in results"""
            self.append_result(
                f"\nMessage {idx + 1} Decryption Info:\n"
                f"Seed: {seed_val}\n"
                f"Keyword: {keyword}\n"
                f"Recipient ID: recipient_{idx}\n"
                "-----------------------\n"
            )

    def process_compressed_messages(self, messages: List[Tuple[str, Optional[int], str]], target_size: int):
            """Updated process_compressed_messages with proper coefficient handling"""
            encryption_system = self.shared_state.get('encryption')
            if not encryption_system:
                self.append_result("No encryption system found. Cannot proceed.\n")
                return

            # Clear old data
            encryption_system.messages.clear()
            encryption_system.perfect_seeds.clear()
            encryption_system.encryption_data.clear()
            self.shared_state.setdefault('pattern_storage', {})

            current_time = int(time.time() * 1000)
            compressed_results = []

            for i, (message_text, provided_seed, keyword) in enumerate(messages):
                self.append_result(f"\nProcessing Message {i + 1}...\n")

                try:
                    msg_bytes = message_text.encode()
                    success, entropy = encryption_system.add_message(msg_bytes, provided_seed, keyword)
                    
                    if not success:
                        self.message_rows[i].entropy_label.configure(text="Entropy: Failed")
                        continue

                    # Get the seed that was actually used
                    used_seed = encryption_system.perfect_seeds[-1]
                    iv, ciphertext, ratio, coeffs = encryption_system.encryption_data[-1]

                    # Create pattern key
                    if not keyword:
                        keyword = ''.join(random.choices(string.ascii_letters, k=8))
                    pattern_key = f"{used_seed}_{keyword}_{hashlib.sha256(msg_bytes).hexdigest()[:8]}"

                    # Store pattern references
                    self.shared_state['pattern_storage'][pattern_key] = {
                        'patterns': [],  # Empty for now since no actual compression
                        'pattern_map': {},
                        'checksum': hashlib.sha256(msg_bytes).hexdigest(),
                        'coefficients': coeffs  # Store coefficients with pattern
                    }

                    # Create segment
                    segment = encryption_system.create_encrypted_segment(used_seed, iv, ciphertext, ratio)
                    if not hasattr(encryption_system, 'combined_data'):
                        encryption_system.combined_data = b''
                    encryption_system.combined_data += segment

                    compressed_results.append({
                        'pattern_key': pattern_key,
                        'seed': used_seed,
                        'keyword': keyword,
                        'verification': hashlib.sha256(msg_bytes).hexdigest(),
                        'timestamp': current_time
                    })

                    self.message_rows[i].entropy_label.configure(text=f"Entropy ~ {ratio:.4f}")
                    self.append_result(
                        f"Using custom seed {used_seed} (entropy ~ {ratio:.10f})\n"
                        f"Compression ratio: {ratio:.2%}\n"
                        f"Pattern key: {pattern_key}\n"
                        f"Keyword used: {keyword}\n"
                    )

                except Exception as e:
                    self.append_result(f"Error processing message {i + 1}: {str(e)}\n")
                    continue

            # Build final identifier
            if compressed_results and hasattr(encryption_system, 'combined_data'):
                combined = encryption_system.combined_data
                identifier = hashlib.sha256(combined).hexdigest()[:target_size]

                self.shared_state.setdefault('encrypted_storage', {})
                self.shared_state['encrypted_storage'][identifier] = {
                    'combined_data': combined,
                    'patterns': compressed_results,
                    'timestamp': current_time
                }

                self.shared_state.setdefault('active_identifiers', set())
                self.shared_state['active_identifiers'].add(identifier)

                # Update encryption state
                if hasattr(encryption_system, 'state'):
                    encryption_system.state['compressed_data'] = {
                        'encrypted_storage': self.shared_state['encrypted_storage'],
                        'pattern_storage': self.shared_state['pattern_storage'],
                        'active_identifiers': self.shared_state['active_identifiers']
                    }
                    encryption_system.save_state()

                # Display identifier
                self.identifier_text.configure(state='normal')
                self.identifier_text.delete("1.0", tk.END)
                self.identifier_text.insert(tk.END, identifier)
                self.identifier_text.configure(state='disabled')

                self.last_identifier = identifier
                self.append_result(
                    f"\nIdentifier ({target_size} chars): {identifier}\n"
                    "Encrypted data stored.\n"
                )

    def add_more_rows(self):
        """Prompt user to add more message rows."""
        num_rows = simpledialog.askinteger("Add Rows", "Number of rows to add:", minvalue=1, maxvalue=20)
        if num_rows:
            parent = self.message_rows[0].frame.master
            current_count = len(self.message_rows)
            for i in range(num_rows):
                self.create_message_row(parent, current_count + i)
            self.update_row_numbers()

    def fill_random_data(self):
        """Fill message entries with random text, leaving seeds/keywords blank."""
        for row in self.message_rows:
            length = random.randint(10, 50)
            random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            row.message_entry.delete(0, tk.END)
            row.message_entry.insert(0, random_text)

            row.seed_entry.delete(0, tk.END)
            row.keyword_entry.delete(0, tk.END)
            row.entropy_label.configure(text="Entropy: -")

        self.identifier_text.configure(state='normal')
        self.identifier_text.delete("1.0", tk.END)
        self.identifier_text.configure(state='disabled')
        self.clear_results()
        self.progress_var.set(0)
        self.update_status("Random data filled.")

    def clear_all(self):
        """Clear all input fields and reset everything."""
        for row in self.message_rows:
            row.message_entry.delete(0, tk.END)
            row.seed_entry.delete(0, tk.END)
            row.keyword_entry.delete(0, tk.END)
            row.entropy_label.configure(text="Entropy: -")

        self.identifier_text.configure(state='normal')
        self.identifier_text.delete("1.0", tk.END)
        self.identifier_text.configure(state='disabled')
        self.clear_results()
        self.progress_var.set(0)
        self.update_status("All fields cleared.")
        self.last_identifier = ""

    def delete_row(self, row: MessageRow):
        """Delete a message row, ensuring at least one row remains."""
        if len(self.message_rows) <= 1:
            messagebox.showwarning("Delete Row", "Cannot remove all rows; need at least one.")
            return
        row.frame.destroy()
        self.message_rows.remove(row)
        self.update_row_numbers()

    def update_row_numbers(self):
        """Renumber rows after adding/removing them."""
        for index, row in enumerate(self.message_rows):
            row.update_label(index)

    def copy_identifier(self):
        """Copy the final identifier to clipboard."""
        identifier = self.identifier_text.get("1.0", tk.END).strip()
        if identifier:
            self.clipboard_clear()
            self.clipboard_append(identifier)
            self.update_status("Identifier copied.")
        else:
            messagebox.showinfo("Copy Identifier", "No identifier to copy.")

    def save_identifier(self):
        """Save the current identifier to file."""
        identifier = self.identifier_text.get("1.0", tk.END).strip()
        if not identifier:
            messagebox.showinfo("Save Identifier", "No identifier to save.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".id",
            filetypes=[
                ("Identifier files", "*.id"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(identifier)
                self.update_status(f"Identifier saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Identifier",
                                     f"Failed to save identifier: {str(e)}")

    def update_status(self, message: str):
        """Update the status label at bottom with current operation."""
        self.status_label.configure(text=message)
        self.update_idletasks()

    def append_result(self, msg: str):
        """Append a line to the results text area."""
        self.results_text.configure(state='normal')
        self.results_text.insert(tk.END, msg)
        self.results_text.see(tk.END)
        self.results_text.configure(state='disabled')

    def clear_results(self):
        """Clear the results text area."""
        self.results_text.configure(state='normal')
        self.results_text.delete("1.0", tk.END)
        self.results_text.configure(state='disabled')


class CompressedDecryptionTab(ttk.Frame):
    """
    Tab for loading an 'identifier' (the result of compression+encryption)
    and decrypting messages from it.
    """

    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        super().__init__(parent)
        self.shared_state = shared_state
        self.helper = EncryptionHelper()
        self.compressor = EnhancedCompressor()
        self.integrator = QuantumStackIntegrator()
        self.file_handler = FileHandler()
        self.secure_storage = create_secure_storage()
        
        # Load references from encryption state if possible
        encryption_system = self.shared_state.get('encryption')
        if encryption_system and hasattr(encryption_system, 'state'):
            comp_data = encryption_system.state.get('compressed_data', {})
            self.shared_state['encrypted_storage'] = comp_data.get('encrypted_storage', {})
            self.shared_state['pattern_storage'] = comp_data.get('pattern_storage', {})
            self.shared_state['active_identifiers'] = comp_data.get('active_identifiers', set())

        self.setup_ui()

    def setup_ui(self):
        """Build the UI for compressed decryption: identifier input, seed/keyword, results."""
        # Identifier input
        identifier_frame = ttk.LabelFrame(self, text="Input Identifier for Decryption", padding=10)
        identifier_frame.pack(fill="x", padx=10, pady=5)

        self.identifier_text = scrolledtext.ScrolledText(identifier_frame, height=3)
        self.identifier_text.pack(fill="x", padx=5, pady=5)

        identifier_controls = ttk.Frame(identifier_frame)
        identifier_controls.pack(fill="x", padx=5, pady=5)
        ttk.Button(identifier_controls, text="Load Identifier",
                   command=self.load_identifier).pack(side="left", padx=5)

        # Decryption details
        decrypt_frame = ttk.LabelFrame(self, text="Decryption Details", padding=10)
        decrypt_frame.pack(fill="x", padx=10, pady=5)

        seed_frame = ttk.Frame(decrypt_frame)
        seed_frame.pack(fill="x", pady=5)
        ttk.Label(seed_frame, text="Seed:").pack(side="left")
        self.seed_entry = ttk.Entry(seed_frame, width=20)
        self.seed_entry.pack(side="left", padx=5)

        keyword_frame = ttk.Frame(decrypt_frame)
        keyword_frame.pack(fill="x", pady=5)
        ttk.Label(keyword_frame, text="Keyword:").pack(side="left")
        self.keyword_entry = ttk.Entry(keyword_frame, width=30)
        self.keyword_entry.pack(side="left", padx=5)

        ttk.Button(decrypt_frame, text="Decrypt & Decompress",
                   command=self.decrypt_and_decompress).pack(pady=10)

        # Results
        results_frame = ttk.LabelFrame(self, text="Decryption Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, state='disabled')
        self.results_text.pack(fill="both", expand=True)

    def decrypt_and_decompress(self):
        """Decrypt the data associated with the identifier using seed & keyword."""
        identifier = self.identifier_text.get("1.0", tk.END).strip()
        if not identifier:
            self.append_result("Please provide an identifier for decryption.\n")
            return

        # Get seed and keyword from the UI
        seed_str = self.seed_entry.get().strip()
        keyword_str = self.keyword_entry.get().strip()

        if not seed_str or not keyword_str:
            self.append_result("Please provide both seed and keyword.\n")
            return

        try:
            seed_val = int(seed_str)
        except ValueError:
            self.append_result("Invalid seed format. Must be an integer.\n")
            return

        # List available .enc files
        import glob
        enc_files = glob.glob("quantum_stack_*.enc")
        if not enc_files:
            self.append_result("No encrypted files found.\n")
            return

        # Select the most recent .enc file
        try:
            enc_file = max(enc_files, key=lambda x: os.path.getmtime(x))
        except ValueError:
            self.append_result("Error accessing encrypted files.\n")
            return

        # Try to decrypt
        decrypted = self.integrator.decrypt_message(
            identifier,
            'recipient_0',  # We try with recipient_0 first
            seed_val,
            keyword_str,
            enc_file
        )

        if decrypted:
            try:
                decoded = decrypted.decode('utf-8')
                self.append_result("Decrypted message:\n")
                self.append_result(f"{decoded}\n")
                return
            except UnicodeDecodeError:
                self.append_result("Message could not be decoded as text.\n")
                return

        # If first attempt failed, try other recipient IDs
        for i in range(1, 5):  # Try recipient_1 through recipient_4
            recipient_id = f'recipient_{i}'
            decrypted = self.integrator.decrypt_message(
                identifier,
                recipient_id,
                seed_val,
                keyword_str,
                enc_file
            )
            
            if decrypted:
                try:
                    decoded = decrypted.decode('utf-8')
                    self.append_result("Decrypted message:\n")
                    self.append_result(f"{decoded}\n")
                    return
                except UnicodeDecodeError:
                    continue

        self.append_result("No message found for the provided seed and keyword.\n")

    def load_identifier(self):
        """Load an identifier from a file into the text field."""
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Identifier files", "*.id"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    identifier = f.read().strip()
                self.identifier_text.delete("1.0", tk.END)
                self.identifier_text.insert(tk.END, identifier)
                self.append_result(f"Identifier loaded from {filename}\n")
            except Exception as e:
                self.append_result(f"Error loading identifier: {str(e)}\n")

    def clear(self):
        """Clear all input fields."""
        self.identifier_text.delete("1.0", tk.END)
        self.seed_entry.delete(0, tk.END)
        self.keyword_entry.delete(0, tk.END)
        self.results_text.configure(state='normal')
        self.results_text.delete("1.0", tk.END)
        self.results_text.configure(state='disabled')

    def append_result(self, message: str):
        """Append a line to the results text area."""
        self.results_text.configure(state='normal')
        self.results_text.insert(tk.END, message)
        self.results_text.see(tk.END)
        self.results_text.configure(state='disabled')