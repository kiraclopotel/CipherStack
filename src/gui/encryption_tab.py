# encryption_tab.py

import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, filedialog
from typing import Dict, Any, List, Optional
import time
import random
import string

from utils.helpers import DataValidator, EncryptionHelper

class EncryptionTab(ttk.Frame):
    """
    A tab that allows users to:
     1) Enter multiple messages and optional seeds
     2) Encrypt them via 'encryption.py' (which uses c_k coefficients)
     3) Stack them into a combined hash
     4) Optionally decrypt the last message using seed + c_k
    """

    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        """
        Args:
            parent: The parent Notebook widget.
            shared_state: A dictionary containing references to the encryption system and other global data.
        """
        super().__init__(parent)
        self.shared_state = shared_state
        self.validator = DataValidator()
        self.helper = EncryptionHelper()

        # Lists to hold references for each row of input
        self.message_entries: List[ttk.Entry] = []
        self.entropy_labels: List[ttk.Label] = []
        self.seed_entries: List[ttk.Entry] = []

        self.processing = False  # Flag to avoid multiple simultaneous processes
        self.last_hash = ""      # Stores the last generated or loaded hash

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Build all the UI elements for message input, hashing, and results."""
        # Frame: message input
        input_frame = ttk.LabelFrame(self, text="Message Input", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        # Create initial 5 rows
        for i in range(5):
            self._create_message_row(input_frame, i)

        # Buttons for row manipulation & random filling
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(buttons_frame, text="Add More Rows",
                   command=self._add_more_rows).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Fill Random",
                   command=self._fill_random_data).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Clear All",
                   command=self._clear_all).pack(side="left", padx=5)

        # Frame: Hash Operations
        hash_frame = ttk.LabelFrame(self, text="Hash Operations", padding=10)
        hash_frame.pack(fill="x", padx=10, pady=5)

        self.hash_text = scrolledtext.ScrolledText(hash_frame, height=3)
        self.hash_text.pack(fill="x", padx=5, pady=5)

        hash_controls = ttk.Frame(hash_frame)
        hash_controls.pack(fill="x", padx=5, pady=5)

        ttk.Button(hash_controls, text="Stack Messages",
                   command=self._stack_messages).pack(side="left", padx=5)
        ttk.Button(hash_controls, text="Copy Hash",
                   command=self._copy_hash).pack(side="left", padx=5)
        ttk.Button(hash_controls, text="Save Hash",
                   command=self._save_hash).pack(side="left", padx=5)
        ttk.Button(hash_controls, text="Load Hash",
                   command=self._load_hash).pack(side="left", padx=5)

        # Decrypt the last message with c_k
        ttk.Button(hash_controls, text="Decrypt Last Message",
                   command=self._decrypt_last_message).pack(side="left", padx=5)

        # Frame: Results & Status
        results_frame = ttk.LabelFrame(self, text="Results & Status", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.results_text.pack(fill="both", expand=True)

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            results_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill="x", padx=5, pady=5)

        # Status Label
        self.status_label = ttk.Label(results_frame, text="Ready")
        self.status_label.pack(pady=5)

    def _create_message_row(self, parent: ttk.Frame, index: int):
        """
        Create one row containing:
         - A label "Message {i+1}:"
         - A text entry for the message
         - A label "Entropy: -"
         - A seed entry (optional)
         - A delete button
        """
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x", pady=2)

        # Label + message entry
        ttk.Label(row_frame, text=f"Message {index + 1}:").pack(side="left")
        msg_entry = ttk.Entry(row_frame, width=50)
        msg_entry.pack(side="left", padx=5)
        self.message_entries.append(msg_entry)

        # Entropy label
        ent_label = ttk.Label(row_frame, text="Entropy: -")
        ent_label.pack(side="left", padx=5)
        self.entropy_labels.append(ent_label)

        # Optional seed
        ttk.Label(row_frame, text="Seed (optional):").pack(side="left", padx=5)
        seed_entry = ttk.Entry(row_frame, width=20)
        seed_entry.pack(side="left", padx=5)
        self.seed_entries.append(seed_entry)

        # Delete row
        del_btn = ttk.Button(row_frame, text="×", width=2,
                             command=lambda: self._delete_row(row_frame, index))
        del_btn.pack(side="right", padx=5)

    def _delete_row(self, frame: ttk.Frame, index: int):
        """Remove the specified row from UI and internal references."""
        frame.destroy()
        self.message_entries.pop(index)
        self.entropy_labels.pop(index)
        self.seed_entries.pop(index)
        self._update_row_numbers()

    def _update_row_numbers(self):
        """Renumber row labels after a deletion."""
        for i, entry in enumerate(self.message_entries):
            row_frame = entry.master
            label = row_frame.winfo_children()[0]  # the label is the 1st child
            label.configure(text=f"Message {i + 1}:")

    def _add_more_rows(self):
        """Prompt user for how many extra rows to add."""
        num_rows = simpledialog.askinteger("Add Rows",
                                           "Number of rows to add:",
                                           minvalue=1, maxvalue=20)
        if num_rows:
            parent = self.message_entries[0].master.master
            current_count = len(self.message_entries)
            for _ in range(num_rows):
                self._create_message_row(parent, current_count)
                current_count += 1

    def _fill_random_data(self):
        """Fill each message entry with random text; seeds left blank."""
        for entry in self.message_entries:
            length = random.randint(10, 50)
            random_text = ''.join(random.choices(
                string.ascii_letters + string.digits, k=length))
            entry.delete(0, tk.END)
            entry.insert(tk.END, random_text)

    # ------------------------------------------------------------------
    # STACK / HASH
    # ------------------------------------------------------------------

    def _stack_messages(self):
        """
        Collect all messages + optional seeds,
        pass them to the encryption system => produce a combined hash,
        and display results.
        """
        if self.processing:
            return

        self.processing = True
        self._clear_results()
        self._update_status("Processing messages...")

        enc_system = self.shared_state.get('encryption')
        if not enc_system:
            self._append_result("Encryption system not found in shared_state.\n")
            self.processing = False
            return

        start_time = time.time()

        # Gather valid messages
        valid_msgs = []
        total_filled = len([m for m in self.message_entries if m.get().strip()])
        count = 0
        for i, (msg_entry, seed_entry) in enumerate(zip(self.message_entries, self.seed_entries)):
            msg_text = msg_entry.get().strip()
            if not msg_text:
                continue

            count += 1
            self.progress_var.set((count / total_filled) * 100)
            self.update()

            seed_str = seed_entry.get().strip()
            seed_val = None
            if seed_str:
                try:
                    seed_val = int(seed_str)
                    if seed_val < 0:
                        raise ValueError
                except ValueError:
                    self._append_result(f"Invalid seed for message {i+1}; auto seed used.\n")

            valid_msgs.append((msg_text, seed_val))

        # Clear out old data in encryption system
        enc_system.messages.clear()
        enc_system.perfect_seeds.clear()
        enc_system.encryption_data.clear()

        # Process each
        for i, (message, seed) in enumerate(valid_msgs):
            self._append_result(f"\nProcessing Message {i+1}: {message}\n")
            success, entropy = enc_system.add_message(message.encode('utf-8'), seed)
            if success:
                self.entropy_labels[i].configure(text=f"Entropy: {entropy:.10f}")
                used_seed = enc_system.perfect_seeds[-1]
                self._append_result(f"Message stored with seed {used_seed}, entropy={entropy:.10f}\n")
            else:
                self.entropy_labels[i].configure(text="Entropy: Failed")
                self._append_result("Failed to achieve perfect entropy.\n")

        # Combine into final hash if any messages were successfully stored
        if enc_system.messages:
            combined = enc_system.combine_messages()
            final_hash = enc_system.format_hash(combined)

            self.hash_text.delete("1.0", tk.END)
            self.hash_text.insert(tk.END, final_hash)
            self.last_hash = final_hash

            self._append_result(f"\nCombined total bytes: {len(combined)}\nFinal Hash: {final_hash}\n")

        elapsed = time.time() - start_time
        self._update_status(f"Processing finished in {elapsed:.2f}s.")
        self.progress_var.set(100)
        self.processing = False

    def _copy_hash(self):
        """Copy the final hash from the text area to clipboard."""
        hash_val = self.hash_text.get("1.0", tk.END).strip()
        if hash_val:
            self.clipboard_clear()
            self.clipboard_append(hash_val)
            self._update_status("Hash copied to clipboard.")

    def _save_hash(self):
        """Save the final hash to a file."""
        hash_val = self.hash_text.get("1.0", tk.END).strip()
        if not hash_val:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".hash",
            filetypes=[
                ("Hash files", "*.hash"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(hash_val)
                self._update_status(f"Hash saved to {filename}")
            except Exception as e:
                self._update_status(f"Error saving hash: {str(e)}")

    def _load_hash(self):
        """Load a hash from a file into the text area."""
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
                    data = f.read().strip()
                self.hash_text.delete("1.0", tk.END)
                self.hash_text.insert(tk.END, data)
                self._update_status(f"Hash loaded from {filename}")
            except Exception as e:
                self._update_status(f"Error loading hash: {str(e)}")

    def _clear_all(self):
        """Clear all message & seed entries, plus results."""
        for entry in self.message_entries:
            entry.delete(0, tk.END)
        for label in self.entropy_labels:
            label.configure(text="Entropy: -")
        for seed_entry in self.seed_entries:
            seed_entry.delete(0, tk.END)

        self.hash_text.delete("1.0", tk.END)
        self.results_text.delete("1.0", tk.END)
        self.progress_var.set(0)
        self._update_status("All fields cleared.")
        self.last_hash = ""

    # ------------------------------------------------------------------
    # DECRYPT with c_k
    # ------------------------------------------------------------------

    def _decrypt_last_message(self):
        """
        Decrypt the last message stored in encryption_data using (seed + c_k).
        This assumes encryption_data is storing tuples of:
          (iv, ciphertext, entropy, coefficients)
        for each message. If any step fails, a message is appended to results.
        """
        enc_system = self.shared_state.get('encryption')
        if not enc_system or not enc_system.messages:
            self._append_result("No messages found to decrypt.\n")
            return

        # We take the last message from encryption_data
        last_idx = len(enc_system.messages) - 1

        try:
            # encryption_data is stored as: (iv, ciphertext, entropy, coeffs)
            iv, ciphertext, _entropy, coeffs = enc_system.encryption_data[last_idx]
            seed_used = enc_system.perfect_seeds[last_idx]

            # Decrypt with seed + c_k
            decrypted_bytes = enc_system.decrypt_with_seed_and_coeffs(ciphertext, seed_used, iv, coeffs)
            dec_text = decrypted_bytes.decode('utf-8', errors='replace')
            self._append_result(f"Decrypted last message:\n{dec_text}\n\n")

        except Exception as e:
            self._append_result(f"Error decrypting last message: {str(e)}\n")

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------

    def _update_status(self, msg: str):
        """Update the status_label with a message."""
        self.status_label.configure(text=msg)
        self.update()

    def _clear_results(self):
        """Clear the results text area."""
        self.results_text.delete("1.0", tk.END)

    def _append_result(self, msg: str):
        """Append text to the results area."""
        self.results_text.insert(tk.END, msg)
        self.results_text.see(tk.END)
