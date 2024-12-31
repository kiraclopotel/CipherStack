# main_window.py

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any
import sys
from pathlib import Path

# Append parent directory to sys.path so we can import modules in a typical project structure
sys.path.append(str(Path(__file__).parent.parent))

from core.encryption import QuantumStackEncryption
from core.timeline import TimelineManager
from core.mathematics import MathematicalCore
from utils.helpers import ConfigManager, MetricsCollector
from gui.encryption_tab import EncryptionTab
from gui.visualization_tab import VisualizationTab
from gui.tests_tab import TestsTab
from gui.decryption_tab import DecryptionTab
from gui.compression_tabs import CompressedEncryptionTab, CompressedDecryptionTab

class MainWindow(tk.Tk):
    """
    The main application window for the Quantum Stack Encryption system.
    It creates and manages the top-level UI, including:
      - The Notebook tabs (Encryption, Decryption, Visualization, Tests, etc.)
      - The main menu bar
      - The status bar
      - Shared state references to core objects (e.g., encryption system)
    """

    def __init__(self):
        super().__init__()

        # 1) Initialize core components
        self.encryption = QuantumStackEncryption()
        self.timeline_manager = TimelineManager()
        self.math_core = MathematicalCore()
        self.config_manager = ConfigManager()
        self.metrics_collector = MetricsCollector()

        # 2) Setup main window geometry/title
        self.title("Encrypt Stack Shield")
        self.geometry("1200x900")

        # 3) Create shared state dictionary
        self.shared_state: Dict[str, Any] = {
            'encryption': self.encryption,
            'timeline': self.timeline_manager,
            'math_core': self.math_core,
            'metrics': self.metrics_collector,
            'config': self.config_manager,
            'pattern_storage': {},
            'encrypted_storage': {}  # for storing compressed/stacked data
        }

        # 4) Setup the UI (menu, notebook, tabs)
        self._setup_ui()

    def _setup_ui(self):
        """
        Build the main UI:
          - Create the menu
          - Create a container + notebook
          - Instantiate each tab and add them to the notebook
          - Create a status bar
        """
        # Create menu bar
        self._create_menu()

        # Create container frame
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        # Create the Notebook
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create each tab
        self.encryption_tab = EncryptionTab(self.notebook, self.shared_state)
        self.decryption_tab = DecryptionTab(self.notebook, self.shared_state)
        self.visualization_tab = VisualizationTab(self.notebook, self.shared_state)
        self.tests_tab = TestsTab(self.notebook, self.shared_state)
        self.compressed_encryption_tab = CompressedEncryptionTab(self.notebook, self.shared_state)
        self.compressed_decryption_tab = CompressedDecryptionTab(self.notebook, self.shared_state)

        # Add tabs to the notebook
        self.notebook.add(self.encryption_tab, text="Encryption")
        self.notebook.add(self.decryption_tab, text="Decryption")
        self.notebook.add(self.visualization_tab, text="Visualization")
        self.notebook.add(self.tests_tab, text="Tests")
        self.notebook.add(self.compressed_encryption_tab, text="Compressed Encryption")
        self.notebook.add(self.compressed_decryption_tab, text="Compressed Decompression")

        # Status bar at the bottom
        self.status_bar = ttk.Label(container, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind tab change events
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    def _create_menu(self):
        """
        Build the top menu bar with 'File', 'Tools', and 'Help' menus,
        each containing relevant commands.
        """
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self._new_session)
        file_menu.add_command(label="Save State", command=self._save_state)
        file_menu.add_command(label="Load State", command=self._load_state)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Clear All", command=self._clear_all)
        tools_menu.add_command(label="Reset Statistics", command=self._reset_stats)
        tools_menu.add_command(label="Export Metrics", command=self._export_metrics)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_docs)
        help_menu.add_command(label="About", command=self._show_about)

    def _on_tab_changed(self, event):
        """
        Triggered whenever the user switches tabs in the Notebook.
        Updates status bar text and refreshes certain tabs if needed.
        """
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, "text")

        # Update status bar
        self.status_bar.config(text=f"Current View: {tab_text}")

        # Refresh if needed
        if tab_text == "Visualization":
            self.visualization_tab.refresh()
        elif tab_text == "Tests":
            self.tests_tab.refresh()

    # ------------------------------------------------------------------
    # FILE MENU ACTIONS
    # ------------------------------------------------------------------

    def _new_session(self):
        """
        Prompt to start a new session, clearing all data.
        """
        if messagebox.askyesno("New Session", "Start a new session? This clears all data."):
            self._clear_all()

    def _save_state(self):
        """
        Save current config to disk.
        If you want to also save encryption or timeline state, do so in your core classes.
        """
        self.config_manager.save_config()
        self.status_bar.config(text="State saved")

    def _load_state(self):
        """
        Load config from disk. 
        (If you want to load encryption or timeline state, do it in your core classes.)
        """
        if self.config_manager.load_config():
            self.status_bar.config(text="State loaded")
        else:
            self.status_bar.config(text="Error loading state")

    # ------------------------------------------------------------------
    # TOOLS MENU ACTIONS
    # ------------------------------------------------------------------

    def _clear_all(self):
        """
        Clears all data from the encryption system, timeline, metrics,
        and resets each tab's UI elements.
        """
        # Re-initialize core objects
        self.encryption = QuantumStackEncryption()
        self.timeline_manager = TimelineManager()
        self.metrics_collector.clear_metrics()

        # Update shared state
        self.shared_state.update({
            'encryption': self.encryption,
            'timeline': self.timeline_manager,
            'pattern_storage': {},
            'encrypted_storage': {}
        })

        # Clear tab UIs
        self.encryption_tab.clear()
        self.decryption_tab.clear()
        self.visualization_tab.clear()
        self.tests_tab.clear()
        self.compressed_encryption_tab.clear_all()
        self.compressed_decryption_tab.clear()

        self.status_bar.config(text="All data cleared")

    def _reset_stats(self):
        """
        Reset runtime statistics and refresh the visualization tab if needed.
        """
        self.metrics_collector.clear_metrics()
        self.visualization_tab.refresh()
        self.status_bar.config(text="Statistics reset")

    def _export_metrics(self):
        """
        Export the collected metrics to a JSON file.
        """
        stats = self.metrics_collector.get_statistics()
        if stats:
            path = Path("metrics_export.json")
            self.metrics_collector.export_to_file(path)
            self.status_bar.config(text=f"Metrics exported to {path}")

    # ------------------------------------------------------------------
    # HELP MENU ACTIONS
    # ------------------------------------------------------------------

    def _show_docs(self):
        """Show a docs reference (dummy text)."""
        messagebox.showinfo("Documentation",
                            "Refer to official documentation for comprehensive info.\n"
                            "Currently no doc link available in this example.")

    def _show_about(self):
        """Show 'About' dialog with version info."""
        about_text = (
            "Encrypt Stack Shield\n"
            "Version 1.0\n\n"
            "An advanced encryption system using quantum-inspired algorithms\n"
            "and mathematical optimization."
        )
        messagebox.showinfo("About", about_text)

    # ------------------------------------------------------------------
    # PUBLIC METHOD
    # ------------------------------------------------------------------

    def run(self):
        """Start the main application loop."""
        self.mainloop()


# If running this file directly:
if __name__ == "__main__":
    app = MainWindow()
    app.run()
