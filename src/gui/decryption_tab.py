# decryption_tab.py

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from typing import Dict, Any, List, Optional
from utils.helpers import EncryptionHelper

class DecryptionTab(ttk.Frame):
    """
    A UI tab that allows loading a combined 'hash' (hex string)
    and decrypting the contained messages with user-provided seeds.
    """

    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        """
        Initialize the DecryptionTab.

        Args:
            parent: The parent Notebook widget.
            shared_state: A shared dictionary containing references to
                          the encryption system and other global data.
        """
        super().__init__(parent)
        self.shared_state = shared_state
        self.helper = EncryptionHelper()

        # Internal list of seeds for decryption attempts
        self.seed_list: List[int] = []

        self._setup_ui()

    def _setup_ui(self):
        """
        Build the UI components:
          - A text area for the combined hash
          - An entry + listbox for seeds
          - A button to decrypt with the selected seeds
          - A text area to display results
        """

        # Frame: Hash Input
        hash_frame = ttk.LabelFrame(self, text="Input Hash for Decryption", padding=10)
        hash_frame.pack(fill="x", padx=10, pady=5)

        self.hash_text = scrolledtext.ScrolledText(hash_frame, height=3)
        self.hash_text.pack(fill="x", padx=5, pady=5)

        hash_controls = ttk.Frame(hash_frame)
        hash_controls.pack(fill="x", padx=5, pady=5)

        ttk.Button(hash_controls, text="Load Hash", command=self.load_hash).pack(side="left", padx=5)

        # Frame: Seed Input
        seed_frame = ttk.LabelFrame(self, text="Input Seed(s) for Decryption", padding=10)
        seed_frame.pack(fill="x", padx=10, pady=5)

        self.seed_entry = ttk.Entry(seed_frame, width=50)
        self.seed_entry.pack(side="left", padx=5, pady=5)

        ttk.Button(seed_frame, text="Add Seed", command=self.add_seed).pack(side="left", padx=5)

        # Listbox for multiple seeds
        self.seed_listbox = tk.Listbox(seed_frame, height=5, selectmode=tk.SINGLE)
        self.seed_listbox.pack(fill="x", padx=5, pady=5)

        # Decrypt button
        ttk.Button(seed_frame, text="Decrypt Selected", command=self.decrypt_selected).pack(side="left", padx=5)

        # Frame: Results
        results_frame = ttk.LabelFrame(self, text="Decryption Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.results_text.pack(fill="both", expand=True)

    def load_hash(self):
        """
        Load a hash (hex string) from a file.
        Commonly called a .hash or .txt file.
        """
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Hash files", "*.hash"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    hash_value = f.read().strip()
                self.hash_text.delete("1.0", tk.END)
                self.hash_text.insert(tk.END, hash_value)
            except Exception as e:
                self._append_result(f"Error loading hash file: {str(e)}\n")

    def add_seed(self):
        """
        Add a user-provided seed to the internal list of seeds for decryption attempts.
        Validates that the seed is an integer.
        """
        seed_str = self.seed_entry.get().strip()
        if seed_str.isdigit():
            seed_val = int(seed_str)
            self.seed_list.append(seed_val)
            self.seed_listbox.insert(tk.END, f"Seed: {seed_val}")
            self.seed_entry.delete(0, tk.END)
        else:
            self._append_result("Invalid seed. Please enter an integer.\n")

    def decrypt_selected(self):
        """
        Use each seed in seed_list to attempt decryption of the loaded hash data.
        Prints results to the 'results_text' area.
        """
        hash_data = self.hash_text.get("1.0", tk.END).strip()
        if not hash_data:
            self._append_result("No hash data to decrypt.\n")
            return

        # Retrieve the encryption system from shared_state
        encryption_system = self.shared_state.get('encryption')
        if not encryption_system:
            self._append_result("Encryption system not found in shared_state.\n")
            return

        # Convert hex -> bytes
        try:
            combined_data = bytes.fromhex(hash_data)
        except ValueError:
            self._append_result("Invalid hash format. Make sure it's valid hex.\n")
            return

        self._append_result("Starting decryption with provided seeds...\n")

        # Attempt decryption for each seed
        for seed in self.seed_list:
            try:
                message, timestamp = encryption_system.extract_message(combined_data, seed)
                if message:
                    decoded = message.decode("utf-8", errors="replace")
                    self._append_result(f"Decrypted (Seed={seed}):\n{decoded}\n\n")
                else:
                    self._append_result(f"Failed to decrypt with Seed={seed}.\n")
            except Exception as e:
                self._append_result(f"Error with Seed={seed}: {str(e)}\n")

    def clear_results(self):
        """
        Clear the output results text and the seed list,
        returning the tab to initial state.
        """
        self.results_text.delete("1.0", tk.END)
        self.seed_listbox.delete(0, tk.END)
        self.seed_list.clear()

    def _append_result(self, msg: str):
        """
        Helper function to append text to results_text area.
        """
        self.results_text.insert(tk.END, msg)
        self.results_text.see(tk.END)
