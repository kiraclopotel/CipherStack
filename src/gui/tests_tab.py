# tests_tab.py

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Dict, Any, List
import time
import numpy as np
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import logging
import random

# Import the QuantumStackEncryption class from core.encryption
from core.encryption import QuantumStackEncryption
from core.encryption import generate_coefficients
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestsTab(ttk.Frame):
    """
    A UI tab dedicated to running various tests, including:
     - Key generation
     - Entropy analysis
     - Timeline verification
     - Layer function checks
     - Mathematical validation
     - Basic Encryption/Decryption checks
     - **Coefficients Encryption** (advanced c_k usage)
     - Performance metrics
     - Randomness tests
     - Seed uniqueness
     - Hash integrity
     - Boundary conditions
     - Error handling
     - Security validations
     - Coefficient consistency
     - Thread safety
     - Integration tests
    """

    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        """
        Args:
            parent: The parent Notebook widget.
            shared_state: A dictionary containing references
                          to encryption system, timeline, math_core, etc.
        """
        super().__init__(parent)
        self.shared_state = shared_state

        # Internal references to track test results and threads
        self.test_results: Dict[str, bool] = {}
        self.current_test = None
        self.test_running = False

        self._setup_ui()

    def _setup_ui(self):
        """
        Build the User Interface (UI) for the application, enabling users to:
        
        - Select and manage tests to run.
        - Execute selected tests and monitor progress.
        - View results in various formats, including structured data, raw text, and visualizations.
        
        The UI is divided into two main panels:
        
        1. **Left Panel**:
            - **Test Selection**: Allows users to select or deselect individual tests or all tests at once.
            - **Controls**: Buttons to run tests, stop ongoing tests, and clear results.
            - **Progress & Status**: Displays a progress bar and current status of test execution.
        
        2. **Right Panel**:
            - **Results Notebook**: Contains three tabs for viewing results:
                - **Structured Results**: Displays results in a tabular `Treeview` with enhanced column configurations and scrollbars.
                - **Test Results**: Shows raw text output of the tests.
                - **Visualization**: Provides graphical representations of the test data.
        """
        # Create main container
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Left panel: test selection, progress, and status
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)

        # Test selection
        selection_frame = ttk.LabelFrame(left_panel, text="Test Selection", padding=10)
        selection_frame.pack(fill="x", padx=5, pady=5)

        # Select/deselect all checkbox
        self.select_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            selection_frame, 
            text="Select/Deselect All",
            variable=self.select_all_var,
            command=self._toggle_all_tests
        ).pack(fill="x")

        # Test list with checkboxes
        self.test_vars = {}
        test_frame = ttk.Frame(selection_frame)
        test_frame.pack(fill="x", pady=5)

        tests = [
            ("Key Generation",        "Tests key generation uniqueness and strength"),
            ("Entropy Analysis",      "Validates entropy calculation and distribution"),
            ("Timeline Verification", "Checks timeline integrity and continuity"),
            ("Layer Functions",       "Tests layer computation and transitions"),
            ("Mathematical Properties","Validates mathematical foundations"),
            ("Encryption/Decryption", "Tests basic encryption functionality"),
            ("Coefficients Encryption","Tests advanced c_k usage for encryption/decryption"),
            ("Performance Metrics",   "Measures operational performance"),
            ("Randomness Tests",      "Statistical tests for randomness"),
            ("Seed Uniqueness",       "Verifies unique seed generation"),
            ("Hash Integrity",        "Tests hash generation and verification"),
            ("Boundary Conditions",   "Tests handling of edge case inputs"),
            ("Error Handling",        "Ensures proper error responses to invalid inputs"),
            ("Security Validations",  "Checks for information leakage and security robustness"),
            ("Coefficient Consistency","Ensures coefficients remain consistent across operations"),
            ("Thread Safety",         "Verifies encryption system's behavior under concurrent access"),
            ("Integration Tests",     "Validates seamless operation of all components together")
        ]

        for test_name, tooltip in tests:
            var = tk.BooleanVar(value=True)
            self.test_vars[test_name] = var
            cb = ttk.Checkbutton(test_frame, text=test_name, variable=var)
            cb.pack(fill="x")
            self._create_tooltip(cb, tooltip)

        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(control_frame, text="Run Selected Tests",
                   command=self._run_selected_tests).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Stop Tests",
                   command=self._stop_tests).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Clear Results",
                   command=self._clear_results).pack(side="left", padx=5)

        # Progress bar and status
        progress_frame = ttk.Frame(left_panel)
        progress_frame.pack(fill="x", padx=5, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).pack()

        # Right panel with results
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=3)  # Give more weight to results panel

        self.results_notebook = ttk.Notebook(right_panel)
        self.results_notebook.pack(fill="both", expand=True)

        # Structured Results Tab
        structured_frame = ttk.Frame(self.results_notebook)
        structured_frame.pack(fill="both", expand=True)
        self.results_notebook.add(structured_frame, text="Structured Results")

        # Create Treeview with horizontal scrollbar
        tree_container = ttk.Frame(structured_frame)
        tree_container.pack(fill="both", expand=True)

        # Configure columns with specific widths
        columns = ("Test Name", "Description", "Status", "Details")
        self.results_tree = ttk.Treeview(tree_container, columns=columns, show='headings', height=20)
        
        # Column configurations
        widths = {"Test Name": 150, "Description": 250, "Status": 80, "Details": 200}
        for col in columns:
            self.results_tree.column(col, width=widths[col], minwidth=50, anchor="w", stretch=False)
            self.results_tree.heading(col, text=col, anchor="w")

        # Add scrollbars
        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self.results_tree.yview)
        hsb = ttk.Scrollbar(tree_container, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Configure grid weights
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        # Bind double-click to show details
        self.results_tree.bind("<Double-1>", self._on_treeview_double_click)

        # Raw Results Tab
        text_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(text_frame, text="Test Results")

        self.results_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
        self.results_text.pack(fill="both", expand=True)

        # Visualization Tab
        viz_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(viz_frame, text="Visualization")

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _create_tooltip(self, widget, text: str):
        """
        Create a small tooltip with the provided text on hover.
        """
        def show_tooltip(event):
            tip_window = tk.Toplevel()
            tip_window.wm_overrideredirect(True)
            tip_window.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tip_window, text=text, background="#ffffe0",
                              relief="solid", borderwidth=1)
            label.pack()

            def hide_tooltip(_event=None):
                tip_window.destroy()

            tip_window.bind("<Leave>", hide_tooltip)
            widget.bind("<Leave>", hide_tooltip)

        widget.bind("<Enter>", show_tooltip)

    def _toggle_all_tests(self):
        """Select or deselect all tests based on the checkbox state."""
        state = self.select_all_var.get()
        for var in self.test_vars.values():
            var.set(state)

    def _run_selected_tests(self):
        """
        Gather selected tests and run them in a separate thread.
        Updates status/progress in real-time.
        """
        if self.test_running:
            return

        selected_tests = [name for name, var in self.test_vars.items() if var.get()]
        if not selected_tests:
            self._update_status("No tests selected.")
            return

        self.test_running = True
        self.progress_var.set(0)
        self._clear_results()
        self._update_status("Running tests...")

        threading.Thread(target=self._run_tests_thread, args=(selected_tests,), daemon=True).start()

    def _run_tests_thread(self, selected_tests: List[str]):
        """
        Actual logic for running tests in the background.
        We track pass/fail and display results in self.results_text,
        then draw a bar chart of results.
        """
        total_tests = len(selected_tests)
        passed_tests = 0
        start_time = time.time()

        for i, test_name in enumerate(selected_tests):
            if not self.test_running:
                break

            # Update progress
            self.current_test = test_name
            self.progress_var.set((i / total_tests) * 100)
            self._update_status(f"Running {test_name}...")

            method_name = f"{test_name.lower().replace(' ', '_').replace('/', '_')}_test"
            test_method = getattr(self, method_name, None)
            if test_method is None:
                self._log_generic_error(test_name, "Test not implemented.")
                continue

            success = test_method()
            self.test_results[test_name] = success
            if success:
                passed_tests += 1

            # Let the UI catch up
            self.after(100)

        duration = time.time() - start_time
        self._show_summary(total_tests, passed_tests, duration)
        self._update_visualization()

        self.test_running = False
        self.current_test = None
        self.progress_var.set(100)
        self._update_status("Testing completed")

    def _stop_tests(self):
        """Stop the currently running tests."""
        if self.test_running:
            self.test_running = False
            self._update_status("Tests stopped by user.")

    def _clear_results(self):
        """Clear all test results and reset progress."""
        self.test_results.clear()
        self.results_text.delete("1.0", tk.END)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.fig.clear()
        self.canvas.draw()
        self.progress_var.set(0)
        self._update_status("Results cleared.")

    def _show_summary(self, total: int, passed: int, duration: float):
        """
        Print a summary of how many tests ran, how many passed,
        and how long it took.
        """
        summary = (
            "\nTest Summary:\n"
            f"Total Tests: {total}\n"
            f"Passed: {passed}\n"
            f"Failed: {total - passed}\n"
            f"Success Rate: {(passed / total) * 100:.1f}%\n"
            f"Duration: {duration:.2f} seconds\n"
        )
        self.results_text.insert(tk.END, summary)
        self.results_text.see(tk.END)

    def _update_visualization(self):
        """
        Render a bar chart for the test results in self.test_results,
        coloring bars green (pass) or red (fail).
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if not self.test_results:
            ax.text(
                0.5, 0.5, "No test results available",
                ha='center', va='center', fontsize=12
            )
            ax.set_axis_off()
        else:
            test_names = list(self.test_results.keys())
            results = [1 if val else 0 for val in self.test_results.values()]

            bars = ax.bar(range(len(test_names)), results, color=['green' if res else 'red' for res in results])
            ax.set_xticks(range(len(test_names)))
            ax.set_xticklabels(test_names, rotation=45, ha="right")
            ax.set_ylim(0, 1.2)
            ax.set_ylabel("Pass (1) / Fail (0)")
            ax.set_title("Test Results Summary")

            for idx, (bar, val) in enumerate(zip(bars, results)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05, 
                        'Pass' if val else 'Fail', ha='center', va='bottom')

        self.fig.tight_layout()
        self.canvas.draw()

    def _update_status(self, msg: str):
        """Update the status label with a message."""
        self.status_var.set(msg)
        self.update()

    def _log_test_result(self, test_name: str, description: str, inputs: Dict[str, Any],
                        expected: Any, actual: Any, status: str, details: str = ""):
        """Format test results in a clean, professional manner"""
        
        # Format the test result in a structured way
        recipe = f"""
    Test: {test_name}
    Status: {status}
    ----------------------------------------
    Description: {description}

    Inputs:
    {json.dumps(inputs, indent=2)}

    Expected:
    {expected}

    Results Summary:
    """

        # Format details based on test type
        if test_name == "Randomness Tests":
            # Parse p-values and statistics
            details_lines = details.split('\n')
            stats = {}
            current_iteration = 0
            iterations = []
            
            for line in details_lines:
                if any(x in line for x in ['p-value', 'Ratio']):
                    key, value = line.split(':')
                    stats[key.strip()] = float(value.strip())
                    if len(stats) == 4:  # Complete set of stats
                        iterations.append(stats)
                        stats = {}
                        current_iteration += 1
            
            # Format statistics summary
            recipe += "Statistical Test Results:\n"
            for i, stats in enumerate(iterations):
                recipe += f"\nIteration {i+1}:\n"
                recipe += f"  Monobit Test:     {stats.get('Monobit p-value', 0):.4f}\n"
                recipe += f"  Runs Test:        {stats.get('Runs p-value', 0):.4f}\n"
                recipe += f"  Chi-squared Test: {stats.get('Chi-squared p-value', 0):.4f}\n"
                recipe += f"  Avalanche Ratio:  {stats.get('Avalanche Ratio', 0):.4f}\n"
            
            # Add overall assessment
            recipe += "\nOverall Assessment:\n"
            all_monobit = [x.get('Monobit p-value', 0) for x in iterations]
            all_runs = [x.get('Runs p-value', 0) for x in iterations]
            all_chi = [x.get('Chi-squared p-value', 0) for x in iterations]
            all_avalanche = [x.get('Avalanche Ratio', 0) for x in iterations]
            
            recipe += f"  Average Monobit p-value:     {np.mean(all_monobit):.4f}\n"
            recipe += f"  Average Runs p-value:        {np.mean(all_runs):.4f}\n"
            recipe += f"  Average Chi-squared p-value: {np.mean(all_chi):.4f}\n"
            recipe += f"  Average Avalanche Ratio:     {np.mean(all_avalanche):.4f}\n"
            
        else:
            # For other tests, display details directly
            recipe += details if details else "No additional details available."

        recipe += "\n----------------------------------------\n"
        
        # Update the results text
        self.results_text.insert(tk.END, recipe)
        
        # Update Treeview with summarized info
        summary = details.split('\n')[0] if details else "See full results for details"
        self.results_tree.insert('', tk.END, values=(test_name, description, status, summary))

    def _log_generic_error(self, test_name: str, error_msg: str):
        """
        Logs a generic error when a test method is not implemented or fails to execute.
        """
        test_description = "No description available."
        inputs = {}
        expected = "Test should run without errors."
        actual = error_msg
        status = "Failed"
        details = error_msg

        self._log_test_result(test_name, test_description, inputs, expected, actual, status, details)

    def _on_treeview_double_click(self, event):
        """
        Handle double-click event on Treeview to show full details in a popup.
        """
        item_id = self.results_tree.focus()
        if not item_id:
            return
        item = self.results_tree.item(item_id)
        test_name, description, status, details_snippet = item['values']
        
        # Fetch full details from the Test Results text
        # This assumes that the details are logged sequentially in the Test Results tab
        # For a more robust solution, consider storing full details separately
        # Here, we'll search for the test name in the results_text
        content = self.results_text.get("1.0", tk.END)
        try:
            start_index = content.index(f"Test Name: {test_name}")
            end_index = content.index("----------------------------------------", start_index)
            full_details = content[start_index:end_index].strip()
        except ValueError:
            full_details = "No additional details available."

        # Create a popup window
        popup = tk.Toplevel()
        popup.title(f"Details for {test_name}")
        popup.geometry("600x400")
        popup.transient(self)
        popup.grab_set()

        text_widget = scrolledtext.ScrolledText(popup, wrap=tk.WORD)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert(tk.END, full_details)
        text_widget.configure(state='disabled')


    # --------------------------------------------------------------------------
    # INDIVIDUAL TEST METHODS
    # --------------------------------------------------------------------------

    def key_generation_test(self) -> bool:
        """
        Validate the uniqueness of generated keys using random seeds.
        """
        test_name = "Key Generation Test"
        description = "Validates that keys generated with random seeds are unique."
        success = True
        inputs = {"number_of_keys": 100, "key_length": 32}
        expected = "All generated keys should be unique."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            keys = set()
            for _ in range(inputs['number_of_keys']):
                seed = random.randint(1, 2**32 - 1)
                key = encryption_system.generate_adaptive_key(seed, inputs['key_length'])
                if key in keys:
                    actual = "Duplicate key found."
                    details = f"Seed causing duplication: {seed}"
                    success = False
                    break
                keys.add(key)
            if success:
                actual = "All keys were unique."
                details = "No duplicates found."
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def entropy_analysis_test(self) -> bool:
        """
        Enhanced Entropy Analysis Test using random messages and multiple iterations.
        """
        test_name = "Entropy Analysis Test"
        description = "Evaluates the entropy of ciphertexts generated from random messages and seeds."
        success = True
        inputs = {"iterations": 20, "message_length_range": (10, 1024)}
        expected = "Entropy values should be within [0, 1] and have low standard deviation."
        actual = ""
        details = ""
        entropies = []
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            for _ in range(inputs['iterations']):
                message_length = random.randint(*inputs['message_length_range'])
                msg = bytes(random.getrandbits(8) for _ in range(message_length))
                seed_result = encryption_system.find_perfect_entropy_seed(msg)
                if not seed_result or seed_result[0] is None:
                    actual = f"Could not find perfect seed for message of length {message_length}."
                    details = "Failed to find perfect seed."
                    success = False
                    break
                seed, iv, ct, entropy = seed_result
                if not (0 <= entropy <= 1):
                    actual = f"Entropy {entropy:.4f} out of range for message of length {message_length}."
                    details = "Entropy out of expected bounds."
                    success = False
                    break
                entropies.append(entropy)

            if success and len(entropies) > 1:
                std_dev = np.std(entropies)
                details += f"\nEntropy Std Dev: {std_dev:.4f}"
                if std_dev > 0.15:
                    actual = "High standard deviation among entropy values."
                    success = False
                else:
                    actual = "Entropy values are within expected range with low standard deviation."
            elif success:
                actual = "Entropy analysis completed with single iteration."
                details += "\nOnly one entropy value computed."

            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def timeline_verification_test(self) -> bool:
        """
        Check that timeline manager can create/verify markers with all required fields.
        """
        test_name = "Timeline Verification Test"
        description = "Ensures timeline markers are correctly created and verified with all necessary fields."
        success = True
        inputs = {"test_message": "Timeline test message", "seed": 12345, "msg_id": 0}
        expected = "Timeline markers should contain all required fields and verify correctly."
        actual = ""
        details = ""
        try:
            timeline: TimelineManager = self.shared_state['timeline']
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']

            test_message = inputs['test_message'].encode()
            seed = inputs['seed']
            msg_id = inputs['msg_id']

            marker = timeline.create_marker(seed, msg_id, test_message, 1.0)
            needed_fields = ['seed', 'id', 'timestamp', 'entropy', 'layer', 'timeline', 'checksum']
            missing_fields = [f for f in needed_fields if f not in marker]
            if missing_fields:
                actual = f"Missing fields in marker: {', '.join(missing_fields)}."
                details = "Marker creation incomplete."
                success = False
            else:
                if not timeline.verify_marker(seed, msg_id, test_message):
                    actual = "Marker verification failed."
                    details = "Verification logic did not confirm the marker."
                    success = False
                else:
                    actual = "Marker verification succeeded."
                    details = "All required fields present and verified."

            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def layer_functions_test(self) -> bool:
        """
        Validate that the core math function 'compute_layer' returns correct digit-based layers.
        """
        test_name = "Layer Functions Test"
        description = "Ensures 'compute_layer' function returns expected layers based on input values."
        success = True
        inputs = {"test_values": [10, 100, 1000, 9999], "expected_layers": [2, 3, 4, 4]}
        expected = "Layer functions should return correct layers for given inputs."
        actual = ""
        details = ""
        try:
            math_core: MathematicalCore = self.shared_state['math_core']
            for val, exp in zip(inputs['test_values'], inputs['expected_layers']):
                got = math_core.compute_layer(val)
                if got != exp:
                    actual += f"Incorrect layer for value {val}: expected {exp}, got {got}.\n"
                    details += f"Value: {val}, Expected: {exp}, Got: {got}\n"
                    success = False
            if success:
                actual = "All layer function outputs are correct."
                details = "No discrepancies found."
            else:
                actual = "One or more layer function outputs are incorrect."
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def mathematical_properties_test(self) -> bool:
        """
        Quick check that layer_function for small n,k doesn't yield negative or nonsense.
        """
        test_name = "Mathematical Properties Test"
        description = "Validates mathematical functions do not return negative or nonsensical values."
        success = True
        inputs = {
            "n_values": list(range(1, 6)),  # Convert range to list
            "k_values": list(range(1, 4))   # Convert range to list
        }
        expected = "Mathematical functions should return non-negative, logical values."
        actual = ""
        details = ""
        try:
            math_core: MathematicalCore = self.shared_state['math_core']
            for n in inputs['n_values']:
                for k in inputs['k_values']:
                    val = math_core.layer_function(float(n), k)
                    if val < 0:
                        actual += f"Negative result for n={n}, k={k}: {val}\n"
                        details += f"n={n}, k={k}, Value={val}\n"
                        success = False
            if success:
                actual = "All mathematical function outputs are valid."
                details = "No negative or nonsensical values found."
            else:
                actual = "One or more mathematical function outputs are invalid."
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def encryption_decryption_test(self) -> bool:
        """
        Basic encryption/decryption test using random seeds and messages.
        """
        test_name = "Encryption/Decryption Test"
        description = "Validates that data encrypted and then decrypted matches the original data."
        success = True
        inputs = {"iterations": 10, "message_size_choices": [16, 64, 256]}
        expected = "Decrypted data should exactly match the original data."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            for _ in range(inputs['iterations']):
                size = random.choice(inputs['message_size_choices'])
                data = bytes(random.getrandbits(8) for _ in range(size))
                seed = random.randint(1, 2**32 - 1)
                iv, ciphertext = encryption_system.encrypt_with_seed(data, seed)
                decrypted = encryption_system.decrypt_with_seed(ciphertext, seed, iv)
                if data != decrypted:
                    actual += f"Mismatch with seed={seed}, size={size}.\n"
                    details += f"Original: {data}\nDecrypted: {decrypted}\n"
                    success = False
            if success:
                actual = "All encryption/decryption cycles matched."
                details = "No mismatches found."
            else:
                actual = "One or more encryption/decryption cycles did not match."
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred during encryption/decryption."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def coefficients_encryption_test(self) -> bool:
        """Test advanced c_k usage with perfect entropy seed + coefficients."""
        test_name = "Coefficients Encryption Test"
        description = "Ensures that coefficients are correctly used during encryption and decryption with perfect entropy seeds."
        success = True
        inputs = {"message": "This is a c_k usage test with perfect entropy seed"}
        expected = "Decrypted message should match the original using coefficients and achieve near-perfect entropy."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            message = inputs['message'].encode()
            
            # Find perfect seed WITHOUT coefficients first
            s_res = encryption_system.find_perfect_entropy_seed(message)
            if s_res[0] is None:
                actual = "Could not find perfect seed."
                details = "find_perfect_entropy_seed failed to find a suitable seed."
                success = False
            else:
                perfect_seed = s_res[0]
                logger.debug(f"Found perfect seed {perfect_seed}, now applying coefficients")

                # Always apply coefficients to the perfect seed  
                coeffs = generate_coefficients(100)
                iv, ct = encryption_system.encrypt_with_seed_and_coeffs(message, perfect_seed, coeffs)
                bits = np.unpackbits(np.frombuffer(ct, dtype=np.uint8))
                final_entropy = encryption_system.calculate_entropy(bits)

                # Decrypt with coefficients
                decrypted = encryption_system.decrypt_with_seed_and_coeffs(ct, perfect_seed, iv, coeffs)
                if decrypted != message:
                    actual = "Decrypted message does not match original."
                    details = "Mismatch after decryption with coefficients."
                    success = False
                else:
                    actual = f"Decrypted message matches original. Entropy: {final_entropy:.4f}"
                    details = f"Used Perfect Seed: {perfect_seed}, Coefficients: {coeffs}"
                    if abs(final_entropy - 1.0) > 0.01:
                        actual += " But entropy is not close to 1.0."
                        details += " Entropy check failed."
                        success = False
                            
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred during coefficients encryption test."
            details = str(e)
            status = "Failed"
            success = False
                
        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def performance_metrics_test(self) -> bool:
        """
        Check encryption/decryption speeds for random messages.
        """
        test_name = "Performance Metrics Test"
        description = "Measures the time taken for encryption and decryption operations."
        success = True
        inputs = {"message_sizes": [16, 64, 256], "iterations": 10}
        expected = "Average encryption and decryption times should be below 1.0 seconds."
        actual = ""
        details = ""
        times_enc = []
        times_dec = []
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            for size in inputs['message_sizes']:
                for _ in range(inputs['iterations']):
                    data = bytes([i % 256 for i in range(size)])
                    seed = random.randint(1, 2**32 - 1)

                    start_enc = time.time()
                    iv, ciphertext = encryption_system.encrypt_with_seed(data, seed)
                    enc_time = time.time() - start_enc
                    times_enc.append(enc_time)

                    start_dec = time.time()
                    decrypted = encryption_system.decrypt_with_seed(ciphertext, seed, iv)
                    dec_time = time.time() - start_dec
                    times_dec.append(dec_time)

                    if data != decrypted:
                        actual += f"Mismatch in encryption/decryption for size={size}, seed={seed}.\n"
                        details += f"Size: {size}, Seed: {seed}\n"
                        success = False

            avg_enc = np.mean(times_enc) if times_enc else 0
            avg_dec = np.mean(times_dec) if times_dec else 0

            actual += (
                f"Average Encryption Time: {avg_enc:.5f}s\n"
                f"Average Decryption Time: {avg_dec:.5f}s\n"
            )
            details += (
                f"Encryption Times: {times_enc}\n"
                f"Decryption Times: {times_dec}\n"
            )

            if avg_enc > 1.0 or avg_dec > 1.0:
                actual += "Some operations exceed 1.0 seconds.\n"
                details += "Performance threshold exceeded."
                success = False

            if success:
                actual += "All encryption/decryption operations are within performance thresholds."
                details += "Performance metrics are satisfactory."
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred during performance metrics test."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def randomness_tests_test(self) -> bool:
        """
        Statistical tests (monobit, runs, chi-squared, avalanche) on ciphertext.
        """
        test_name = "Randomness Tests"
        description = "Performs statistical randomness tests on ciphertext outputs."
        success = True
        inputs = {"iterations": 25, "message_length": 25000}
        expected = "Ciphertext should exhibit high randomness characteristics."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            for _ in range(inputs['iterations']):
                message = bytes(random.getrandbits(8) for _ in range(inputs['message_length']))
                seed = random.randint(1, 2**32 - 1)
                iv, ciphertext = encryption_system.encrypt_with_seed(message, seed)
                bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))

                # Monobit Test
                mono_p = encryption_system.monobit_test(bits)
                actual += f"Monobit p-value: {mono_p:.4f}\n"
                details += f"Monobit p-value: {mono_p:.4f}\n"
                if mono_p < 0.01:
                    actual += "Monobit test failed.\n"
                    details += "Ciphertext does not pass Monobit test.\n"
                    success = False

                # Runs Test
                runs_p = encryption_system.runs_test(bits)
                actual += f"Runs p-value: {runs_p:.4f}\n"
                details += f"Runs p-value: {runs_p:.4f}\n"
                if runs_p < 0.01:
                    actual += "Runs test failed.\n"
                    details += "Ciphertext does not pass Runs test.\n"
                    success = False

                # Chi-squared Test
                chi_p = encryption_system.chi_squared_test(bits)
                actual += f"Chi-squared p-value: {chi_p:.4f}\n"
                details += f"Chi-squared p-value: {chi_p:.4f}\n"
                if chi_p < 0.01:
                    actual += "Chi-squared test failed.\n"
                    details += "Ciphertext does not pass Chi-squared test.\n"
                    success = False

                # Avalanche Test
                avalanche_ratio = encryption_system.avalanche_test(message, seed)
                actual += f"Avalanche Ratio: {avalanche_ratio:.4f}\n"
                details += f"Avalanche Ratio: {avalanche_ratio:.4f}\n"
                if avalanche_ratio < 0.4:
                    actual += "Avalanche test failed.\n"
                    details += "Ciphertext does not pass Avalanche test.\n"
                    success = False

            if success:
                actual += "All randomness tests passed."
                details += "Ciphertext exhibits high randomness."
            else:
                actual += "One or more randomness tests failed."
                details += "Ciphertext fails some randomness criteria."
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred during randomness tests."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def seed_uniqueness_test(self) -> bool:
        """
        Ensure find_perfect_entropy_seed gives unique seeds for different messages.
        """
        test_name = "Seed Uniqueness Test"
        description = "Verifies that different messages receive unique seeds."
        success = True
        # Convert bytes to strings for JSON serialization
        inputs = {"messages": ["MsgA", "MsgB", "MsgC", "MsgD", "MsgE"]}  
        expected = "Each message should have a unique perfect entropy seed."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            used_seeds = set()
            for msg in inputs['messages']:
                # Convert string to bytes for encryption
                msg_bytes = msg.encode('utf-8')  
                s_res = encryption_system.find_perfect_entropy_seed(msg_bytes)
                if not s_res or s_res[0] is None:
                    actual += f"No perfect seed found for message: {msg}\n"
                    details += f"Message: {msg}\n"
                    success = False
                    continue

                seed = s_res[0]
                if seed in used_seeds:
                    actual += f"Duplicate seed {seed} found for message: {msg}\n"
                    details += f"Seed: {seed}\n"
                    success = False
                    break
                used_seeds.add(seed)

            if success:
                actual = "All messages have unique seeds."
                details = "No duplicate seeds found."
            else:
                actual = "Seed uniqueness test failed."
                details = "Duplicate seeds detected."
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred during seed uniqueness test."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def hash_integrity_test(self) -> bool:
        """
        Add multiple messages, combine them, generate a hash, then verify it.
        """
        test_name = "Hash Integrity Test"
        description = "Ensures that hashes are correctly generated and verified from combined messages."
        success = True
        # Use strings instead of bytes for JSON serialization
        inputs = {"messages": ["Test message 1", "Another test message", "Final test message"]}
        expected = "Generated hash should match verification criteria."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            for msg in inputs['messages']:
                # Convert to bytes for encryption
                msg_bytes = msg.encode('utf-8')
                ok, _ent = encryption_system.add_message(msg_bytes)
                if not ok:
                    actual += f"Could not add message: {msg}\n"
                    details += f"Message: {msg}\n"
                    success = False

            if success and encryption_system.messages:
                combined = encryption_system.combine_messages()
                if combined:
                    hash_val = encryption_system.format_hash(combined)
                    if not encryption_system.verify_hash(hash_val):
                        actual += "Hash verification failed.\n"
                        details += "Computed hash does not verify correctly.\n"
                        success = False
                    else:
                        actual += "Hash verification succeeded.\n"
                        details += f"Hash Value: {hash_val}\n"
                else:
                    actual += "No combined data to generate hash.\n"
                    details += "Combined data is empty.\n"
                    success = False
            else:
                actual += "Not enough messages for hash test.\n"
                details += "Messages could not be added successfully.\n"
                success = False

            if success:
                actual += "Hash integrity test passed."
                details += "Hashes are correctly generated and verified."
            else:
                actual += "Hash integrity test failed."
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred during hash integrity test."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def boundary_conditions_test(self) -> bool:
        """
        Test encryption/decryption with boundary conditions like empty data and maximum size.
        """
        test_name = "Boundary Conditions Test"
        description = "Validates system's handling of edge case inputs such as empty data and maximum allowed sizes."
        success = True
        # Store descriptions of test cases instead of actual bytes
        inputs = {"test_cases": [
            {"description": "Empty data", "data_type": "empty"},
            {"description": "1MB of zero bytes", "data_type": "zeros", "size": 1024*1024},
            {"description": "10 bytes of 0xFF", "data_type": "ones", "size": 10},
            {"description": "512 random bytes", "data_type": "random", "size": 512}
        ]}
        expected = "System should correctly handle all boundary condition inputs without errors."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            for case in inputs['test_cases']:
                # Generate the test data based on description
                if case['data_type'] == "empty":
                    data = b""
                elif case['data_type'] == "zeros":
                    data = bytes([0] * case['size'])
                elif case['data_type'] == "ones":
                    data = bytes([255] * case['size'])
                else:  # random
                    data = bytes([random.randint(0, 255) for _ in range(case['size'])])

                seed = random.randint(1, 2**32 - 1)
                iv, ciphertext = encryption_system.encrypt_with_seed(data, seed)
                decrypted = encryption_system.decrypt_with_seed(ciphertext, seed, iv)
                if data != decrypted:
                    actual += f"Mismatch for {case['description']}.\n"
                    details += f"Description: {case['description']}\nSeed: {seed}\n"
                    success = False
                
            if success:
                actual = "All boundary conditions handled correctly."
                details = "No issues found with edge case inputs."
            else:
                actual = "One or more boundary conditions failed."
                details = "Issues detected with handling edge case inputs."
            status = "Passed" if success else "Failed"
            
        except Exception as e:
            actual = "Exception occurred during boundary conditions test."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def error_handling_test(self) -> bool:
        """
        Test how the system handles invalid inputs, such as wrong seeds or corrupted ciphertext.
        """
        test_name = "Error Handling Test"
        description = "Ensures the system appropriately handles invalid inputs and scenarios."
        success = True
        inputs = {"error_cases": [
            {"description": "Wrong seed during decryption", "type": "wrong_seed"},
            {"description": "Corrupted ciphertext", "type": "corrupt_ciphertext"},
            {"description": "Invalid seed type", "type": "invalid_seed_type"},
            {"description": "Non-byte message", "type": "non_byte_message"}
        ]}
        expected = "System should raise appropriate exceptions or handle errors gracefully."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            # Case 1: Wrong seed during decryption
            data = b"Valid data"
            seed = random.randint(1, 2**32 - 1)
            iv, ciphertext = encryption_system.encrypt_with_seed(data, seed)
            wrong_seed = seed + 1  # Assuming this creates an invalid seed

            try:
                encryption_system.decrypt_with_seed(ciphertext, wrong_seed, iv)
                actual += "Failed to catch wrong seed during decryption.\n"
                details += "Decryption did not raise exception with wrong seed.\n"
                success = False
            except Exception:
                actual += "Passed: Wrong seed detected during decryption.\n"
                details += "Exception correctly raised for wrong seed.\n"

            # Case 2: Corrupted ciphertext
            corrupted_ciphertext = bytearray(ciphertext)
            if len(corrupted_ciphertext) > 0:
                corrupted_ciphertext[0] ^= 0xFF  # Flip bits of first byte
            try:
                encryption_system.decrypt_with_seed(bytes(corrupted_ciphertext), seed, iv)
                actual += "Failed to catch corrupted ciphertext.\n"
                details += "Decryption did not raise exception with corrupted ciphertext.\n"
                success = False
            except Exception:
                actual += "Passed: Corrupted ciphertext detected during decryption.\n"
                details += "Exception correctly raised for corrupted ciphertext.\n"

            # Case 3: Invalid seed type
            invalid_seed = "invalid_seed"
            try:
                encryption_system.decrypt_with_seed(ciphertext, invalid_seed, iv)
                actual += "Failed to catch invalid seed type.\n"
                details += "Decryption did not raise exception with invalid seed type.\n"
                success = False
            except Exception:
                actual += "Passed: Invalid seed type detected during decryption.\n"
                details += "Exception correctly raised for invalid seed type.\n"

            # Case 4: Non-byte message
            try:
                encryption_system.encrypt_with_seed("This is a string, not bytes", seed)
                actual += "Failed to catch non-byte message during encryption.\n"
                details += "Encryption did not raise exception for non-byte message.\n"
                success = False
            except Exception:
                actual += "Passed: Non-byte message detected during encryption.\n"
                details += "Exception correctly raised for non-byte message.\n"

            if success:
                actual += "All error handling tests passed."
                details += "System correctly handled all invalid input scenarios."
            else:
                actual += "One or more error handling tests failed."
                details += "Issues detected in error handling mechanisms."
            status = "Passed" if success else "Failed"

        except Exception as e:
            actual = "Exception occurred during error handling test."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def security_validation_test(self) -> bool:
        """
        Perform security validations like ensuring ciphertexts do not leak information about plaintexts.
        """
        test_name = "Security Validation Test"
        description = "Ensures that ciphertexts do not leak information about plaintexts."
        success = True
        inputs = {"iterations": 10, "message_length": 64}
        expected = "Ciphertexts should be unique and not reveal any patterns about plaintexts."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            ciphertexts = set()
            for _ in range(inputs['iterations']):
                message = bytes(random.getrandbits(8) for _ in range(inputs['message_length']))
                seed = random.randint(1, 2**32 - 1)
                iv, ciphertext = encryption_system.encrypt_with_seed(message, seed)
                if ciphertext in ciphertexts:
                    actual += "Duplicate ciphertext found for different plaintexts.\n"
                    details += f"Seed: {seed}\nMessage: {message}\n"
                    success = False
                    break
                ciphertexts.add(ciphertext)
            if success:
                actual += "All ciphertexts are unique for different plaintexts.\n"
                details += "No duplicate ciphertexts detected."
            else:
                actual += "Duplicate ciphertext detected, indicating potential information leakage.\n"
                details += "Ciphertext duplication found."
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred during security validation test."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def coefficient_consistency_test(self) -> bool:
        """
        Ensure that coefficients used during encryption are consistent during decryption.
        """
        test_name = "Coefficient Consistency Test"
        description = "Validates that coefficients used in encryption remain consistent during decryption."
        success = True
        inputs = {"message": "Coefficient consistency test message", "seed": None}  # Seed will be random
        expected = "Decrypted message should match the original message using consistent coefficients."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']
            message = inputs['message'].encode()
            seed = random.randint(1, 2**32 - 1)
            inputs['seed'] = seed

            # Encrypt the message with coefficients
            success_add, entropy = encryption_system.add_message(message, seed=seed)
            if not success_add:
                actual = "add_message failed."
                details = "Encryption with coefficients was unsuccessful."
                success = False
            else:
                idx = len(encryption_system.encryption_data) - 1
                if idx < 0:
                    actual = "No encryption data found after add_message."
                    details = "Encryption data index out of range."
                    success = False
                else:
                    (iv, ciphertext, stored_entropy, coeffs) = encryption_system.encryption_data[idx]

                    # Decrypt with coefficients
                    decrypted = encryption_system.decrypt_with_seed_and_coeffs(ciphertext, seed, iv, coeffs)
                    if decrypted != message:
                        actual = "Decrypted message does not match original."
                        details = "Mismatch after decryption with coefficients."
                        success = False
                    else:
                        actual = f"Decrypted message matches original. Entropy: {entropy:.4f}"
                        details = f"Used Seed: {seed}, Coefficients: {coeffs}"
            status = "Passed" if success else "Failed"
        except Exception as e:
            actual = "Exception occurred during coefficient consistency test."
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def thread_safety_test(self) -> bool:
        """
        Ensure that the encryption system is thread-safe by performing concurrent encryptions and decryptions.
        """
        test_name = "Thread Safety Test"
        description = "Verifies that the encryption system behaves correctly under concurrent access."
        success = True
        inputs = {"threads": 5, "operations_per_thread": 50}
        expected = "Encryption system should handle concurrent operations without data corruption or exceptions."
        actual = ""
        details = ""
        try:
            encryption_system: QuantumStackEncryption = self.shared_state['encryption']

            def encrypt_decrypt_operations(thread_id: int):
                for _ in range(inputs['operations_per_thread']):
                    data = bytes(random.getrandbits(8) for _ in range(64))
                    seed = random.randint(1, 2**32 - 1)
                    iv, ciphertext = encryption_system.encrypt_with_seed(data, seed)
                    decrypted = encryption_system.decrypt_with_seed(ciphertext, seed, iv)
                    if data != decrypted:
                        raise ValueError(f"Data mismatch in thread {thread_id}.")

            threads = [threading.Thread(target=encrypt_decrypt_operations, args=(i,)) for i in range(inputs['threads'])]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            actual = "All threads completed without data mismatches."
            details = "Encryption system is thread-safe."
        except Exception as e:
            actual = f"Thread safety test failed: {str(e)}"
            details = str(e)
            success = False
        status = "Passed" if success else "Failed"

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success

    def security_validations_test(self) -> bool:
        """Perform security validations like ensuring ciphertexts do not leak information."""
        test_name = "Security Validations"
        description = "Ensures ciphertexts do not leak information about plaintexts."
        success = True
        inputs = {
            "test_messages": ["SecureMessage1", "SecureMessage2", "SecureMessage3"],
            "iterations": 3
        }
        expected = "Ciphertexts should be unique and not reveal patterns."
        actual = ""
        details = ""

        try:
            encryption_system = self.shared_state['encryption']
            ciphertexts = set()
            
            for msg in inputs['test_messages']:
                msg_bytes = msg.encode('utf-8')
                for _ in range(inputs['iterations']):
                    seed = random.randint(1, 2**32 - 1)
                    iv, ciphertext = encryption_system.encrypt_with_seed(msg_bytes, seed)
                    
                    # Check uniqueness
                    if ciphertext in ciphertexts:
                        actual += f"Duplicate ciphertext found for message: {msg}\n"
                        details += f"Seed: {seed}\n"
                        success = False
                        break
                    ciphertexts.add(ciphertext)
                    
                    # Verify decryption
                    decrypted = encryption_system.decrypt_with_seed(ciphertext, seed, iv)
                    if decrypted != msg_bytes:
                        actual += f"Decryption mismatch for message: {msg}\n"
                        success = False
                        break

            if success:
                actual = "Security validation passed."
                details = "No information leakage detected."
            status = "Passed" if success else "Failed"
            
        except Exception as e:
            actual = f"Security validation failed: {str(e)}"
            details = str(e)
            status = "Failed"
            success = False

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success


        
    def integration_tests_test(self) -> bool:
        """Test complete system integration including all components."""
        test_name = "Integration Tests"
        description = "Tests complete system workflow including encryption, timeline, and coefficient handling."
        success = True
        inputs = {
            "test_messages": [
                "Integration test message 1",
                "Integration test message 2",
                "Integration test with special chars !@#$"
            ],
            "test_operations": [
                "encryption",
                "coefficient_handling",
                "message_extraction",
                "timeline_verification"
            ]
        }
        expected = "All system components should work together seamlessly"
        actual = ""
        details = ""

        try:
            encryption_system = self.shared_state.get('encryption')
            timeline = self.shared_state.get('timeline')

            if not encryption_system or not timeline:
                raise ValueError("Required system components not found")

            # Clear any previous state
            encryption_system.messages.clear()
            encryption_system.perfect_seeds.clear()
            encryption_system.encryption_data.clear()

            # Test each message through the complete workflow
            for idx, msg in enumerate(inputs["test_messages"]):
                details += f"\nTesting message {idx + 1}:\n"
                msg_bytes = msg.encode('utf-8')

                # 1. First process the message with coefficients
                success_add, entropy = encryption_system.add_message(msg_bytes)
                if not success_add:
                    raise ValueError(f"Failed to add message {idx + 1}")

                used_seed = encryption_system.perfect_seeds[-1]
                details += f"- Used seed: {used_seed}\n"
                details += f"- Achieved entropy: {entropy:.6f}\n"

                # 2. Verify coefficients were stored
                entry_data = encryption_system.encryption_data[-1]
                if len(entry_data) != 4 or entry_data[3] is None:
                    raise ValueError(f"Coefficients not properly stored for message {idx + 1}")
                details += f"- Coefficients stored successfully\n"

                # 3. Timeline verification
                marker = timeline.create_marker(used_seed, idx, msg_bytes, entropy)
                if not timeline.verify_marker(used_seed, idx, msg_bytes):
                    raise ValueError(f"Timeline verification failed for message {idx + 1}")
                details += "- Timeline marker verified\n"

                # 4. Create combined data and verify hash
                combined = encryption_system.combine_messages()
                hash_val = encryption_system.format_hash(combined)
                if not encryption_system.verify_hash(hash_val):
                    raise ValueError(f"Hash verification failed for message {idx + 1}")
                details += f"- Hash verified: {hash_val[:32]}...\n"

                # 5. Extract and verify message
                msg_entry = encryption_system.encryption_data[-1]
                if msg_entry:
                    extracted, _ = encryption_system.extract_message(combined, used_seed)
                    if extracted is None or extracted != msg_bytes:
                        raise ValueError(f"Message extraction failed for message {idx + 1}")
                    details += "- Message successfully extracted and verified\n"
                else:
                    raise ValueError(f"No encryption data found for message {idx + 1}")

            actual = "All integration tests passed successfully"
            status = "Passed"
            success = True

        except Exception as e:
            actual = f"Integration test failed: {str(e)}"
            status = "Failed"
            success = False
            if not details:
                details = str(e)

        self._log_test_result(test_name, description, inputs, expected, actual, status, details)
        return success    
        

    # --------------------------------------------------------------------------
    # PUBLIC METHOD: Refresh
    # --------------------------------------------------------------------------

    def refresh(self):
        """
        Called when user switches to 'Tests' tab.
        If we have existing results, re-draw the summary visualization.
        """
        if self.test_results:
            self._update_visualization()