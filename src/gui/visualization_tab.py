# visualization_tab.py

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import time

class VisualizationTab(ttk.Frame):
    """
    A UI tab responsible for visualizing various aspects of the
    encryption system, including combined views, entropy analysis,
    timeline analysis, and a layer/size view.
    """

    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        """
        Args:
            parent: The parent Notebook widget.
            shared_state: A dictionary containing references to the encryption system,
                          timeline data, and other shared objects.
        """
        super().__init__(parent)
        self.shared_state = shared_state

        # We store multiple matplotlib Figures and their corresponding Canvas objects,
        # so that we can easily switch between them (combined, entropy, timeline, etc.).
        self.figures: Dict[str, Figure] = {}
        self.canvases: Dict[str, FigureCanvasTkAgg] = {}

        # Track the current view (e.g. "combined", "entropy")
        self.current_view = "combined"

        # Auto-refresh toggle
        self.auto_refresh = tk.BooleanVar(value=True)

        # Build and lay out the UI
        self._setup_ui()

    def _setup_ui(self):
        """
        Construct the UI:
          - A row of controls at the top (view selector, auto-refresh,
            refresh button, save plot, reset view)
          - A Notebook for different subplots (combined, entropy, timeline, layer)
          - A status label at the bottom
        """
        # Frame of controls
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill="x", padx=5, pady=5)

        # View Selection
        ttk.Label(controls_frame, text="View:").pack(side="left", padx=5)
        self.view_var = tk.StringVar(value="combined")
        views_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.view_var,
            values=["combined", "entropy", "timeline", "layers"],
            state="readonly"
        )
        views_combo.pack(side="left", padx=5)
        views_combo.bind('<<ComboboxSelected>>', self._on_view_changed)

        # Auto-refresh Checkbutton
        ttk.Checkbutton(
            controls_frame,
            text="Auto Refresh",
            variable=self.auto_refresh
        ).pack(side="left", padx=10)

        # Control Buttons
        ttk.Button(controls_frame, text="Refresh", command=self.refresh).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Save Plot", command=self._save_current_plot).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Reset View", command=self._reset_view).pack(side="left", padx=5)

        # Notebook for sub-views
        self.viz_notebook = ttk.Notebook(self)
        self.viz_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create sub-tabs (combined, entropy, timeline, layers)
        self._create_combined_view()
        self._create_entropy_view()
        self._create_timeline_view()
        self._create_layer_view()

        # Status label at the bottom
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var).pack(fill="x", padx=5)

    # ---------------------------------------------------------------------
    # Sub-tab creation
    # ---------------------------------------------------------------------

    def _create_combined_view(self):
        """Setup a 'Combined View' figure and canvas inside the Notebook."""
        frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(frame, text="Combined View")

        fig = Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.figures['combined'] = fig
        self.canvases['combined'] = canvas

    def _create_entropy_view(self):
        """Setup an 'Entropy Analysis' figure and canvas inside the Notebook."""
        frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(frame, text="Entropy Analysis")

        fig = Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.figures['entropy'] = fig
        self.canvases['entropy'] = canvas

    def _create_timeline_view(self):
        """Setup a 'Timeline Analysis' figure and canvas inside the Notebook."""
        frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(frame, text="Timeline Analysis")

        fig = Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.figures['timeline'] = fig
        self.canvases['timeline'] = canvas

    def _create_layer_view(self):
        """Setup a 'Layer Analysis' figure and canvas inside the Notebook."""
        frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(frame, text="Layer Analysis")

        fig = Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.figures['layers'] = fig
        self.canvases['layers'] = canvas

    # ---------------------------------------------------------------------
    # Update each sub-view
    # ---------------------------------------------------------------------

    def _update_combined_view(self):
        """Update the 'Combined View' figure with multiple subplots."""
        fig = self.figures['combined']
        fig.clear()
        gs = GridSpec(2, 2, figure=fig)

        encryption_sys = self.shared_state['encryption']
        if not encryption_sys.messages:
            self._show_no_data(fig)
            return

        # Grab some relevant data
        entropy_data = []
        for entry in encryption_sys.encryption_data:
            # entry is typically (iv, ciphertext, entropy, [coeffs]) => 3rd index is entropy
            if len(entry) >= 3:
                entropy_data.append(entry[2])

        timeline_data = encryption_sys.timeline.get_visualization_data()

        # Subplot 1: Entropy
        ax1 = fig.add_subplot(gs[0, 0])
        if entropy_data:
            ax1.plot(entropy_data, 'b-o', label="Entropy")
            ax1.set_title("Entropy over Messages")
            ax1.set_xlabel("Message Index")
            ax1.set_ylabel("Entropy")
            ax1.grid(True)
        else:
            ax1.text(0.5, 0.5, "No Entropy Data", ha='center', va='center')
            ax1.set_axis_off()

        # Subplot 2: Timeline scatter
        ax2 = fig.add_subplot(gs[0, 1])
        if timeline_data['timestamps']:
            ax2.scatter(timeline_data['timestamps'], range(len(timeline_data['timestamps'])),
                        c=timeline_data['layers'], cmap='viridis', s=50)
            ax2.set_title("Timeline Scatter")
            ax2.set_xlabel("Timestamp")
            ax2.set_ylabel("Message Index")
        else:
            ax2.text(0.5, 0.5, "No Timeline Data", ha='center', va='center')
            ax2.set_axis_off()

        # Subplot 3: Distribution of layers
        ax3 = fig.add_subplot(gs[1, 0])
        if timeline_data['layers']:
            ax3.hist(timeline_data['layers'], bins='auto', alpha=0.7, color='green')
            ax3.set_title("Layer Distribution")
            ax3.set_xlabel("Layer")
            ax3.set_ylabel("Count")
        else:
            ax3.text(0.5, 0.5, "No Layer Data", ha='center', va='center')
            ax3.set_axis_off()

        # Subplot 4: Entropy difference histogram
        ax4 = fig.add_subplot(gs[1, 1])
        if len(entropy_data) > 1:
            diffs = np.diff(entropy_data)
            ax4.hist(diffs, bins='auto', alpha=0.7, color='orange')
            ax4.set_title("Entropy Differences")
            ax4.set_xlabel("Entropy Change")
            ax4.set_ylabel("Frequency")
        else:
            ax4.text(0.5, 0.5, "Not Enough Entropy Data", ha='center', va='center')
            ax4.set_axis_off()

        fig.tight_layout()
        self.canvases['combined'].draw()

    def _update_entropy_view(self):
        """
        Focused view showing more detail about entropy data:
         - A line plot of entropy over time
         - A small 2D 'heatmap' of recent entropies
        """
        fig = self.figures['entropy']
        fig.clear()
        gs = GridSpec(2, 1, figure=fig)

        encryption_sys = self.shared_state['encryption']
        if not encryption_sys.messages:
            self._show_no_data(fig)
            return

        # Extract entropy
        entropy_data = []
        for entry in encryption_sys.encryption_data:
            if len(entry) >= 3:
                entropy_data.append(entry[2])

        # Plot 1: Entropy over messages
        ax1 = fig.add_subplot(gs[0])
        if entropy_data:
            ax1.plot(entropy_data, color='purple', marker='o', label="Entropy")
            ax1.fill_between(range(len(entropy_data)), entropy_data, alpha=0.2, color='purple')
            ax1.set_title("Entropy Over Messages")
            ax1.set_xlabel("Message Index")
            ax1.set_ylabel("Entropy")
            ax1.grid(True)
        else:
            ax1.text(0.5, 0.5, "No Entropy Data", ha='center', va='center')
            ax1.set_axis_off()

        # Plot 2: Heatmap of recent entropies
        ax2 = fig.add_subplot(gs[1])
        if entropy_data:
            data_arr = np.array(entropy_data[-100:])  # last 100
            num_elem = len(data_arr)
            matrix_size = min(10, num_elem)

            # number of rows
            rows = int(np.ceil(num_elem / matrix_size))
            pad_needed = rows * matrix_size - num_elem
            if pad_needed > 0:
                data_arr = np.pad(data_arr, (0, pad_needed), mode='edge')

            ent_mat = data_arr.reshape(rows, matrix_size)
            im = ax2.imshow(ent_mat.T, aspect='auto', cmap='plasma', interpolation='nearest')
            ax2.set_title("Recent Entropy Heatmap")
            ax2.set_xlabel("Block Index")
            ax2.set_ylabel("Entropy Index")
            fig.colorbar(im, ax=ax2, label="Entropy")

        fig.tight_layout()
        self.canvases['entropy'].draw()

    def _update_timeline_view(self):
        """
        Show a timeline scatter + some timeline stats (bar chart, etc.).
        """
        fig = self.figures['timeline']
        fig.clear()
        gs = GridSpec(2, 1, figure=fig)

        timeline = self.shared_state['encryption'].timeline
        data = timeline.get_visualization_data()

        if not data['timestamps']:
            self._show_no_data(fig)
            return

        # Subplot 1: A scatter of timestamps vs. message index
        ax1 = fig.add_subplot(gs[0])
        scatter = ax1.scatter(
            data['timestamps'],
            range(len(data['timestamps'])),
            c=data['layers'],
            cmap='viridis',
            s=60
        )
        ax1.set_title("Timeline Scatter")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Message Index")
        fig.colorbar(scatter, ax=ax1, label="Layer")

        # Subplot 2: Some timeline statistics
        ax2 = fig.add_subplot(gs[1])
        stats = timeline.get_layer_statistics()
        if stats:
            keys = list(stats.keys())
            vals = list(stats.values())
            ax2.bar(keys, vals, alpha=0.7, color='cornflowerblue')
            ax2.set_title("Timeline Layer Stats")
            ax2.set_xticks(range(len(keys)))
            ax2.set_xticklabels(keys, rotation=45)
        else:
            ax2.text(0.5, 0.5, "No Timeline Stats", ha='center', va='center')
            ax2.set_axis_off()

        fig.tight_layout()
        self.canvases['timeline'].draw()

    def _update_layer_view(self):
        """
        This could highlight 'layer function' or 'message size' analysis
        if you want it to. For now, we show a placeholder or a simple approach.
        """
        fig = self.figures['layers']
        fig.clear()
        gs = GridSpec(2, 2, figure=fig)

        timeline = self.shared_state['encryption'].timeline
        data = timeline.get_visualization_data()

        if not data['layers']:
            self._show_no_data(fig)
            return

        # For demonstration, let's assume "layers" can also represent message sizes:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(data['layers'], bins='auto', alpha=0.6, color='teal')
        ax1.set_title("Layer/Size Distribution")
        ax1.set_xlabel("Layer/Size")
        ax1.set_ylabel("Count")

        # Plot size changes over index
        ax2 = fig.add_subplot(gs[0, 1])
        changes = np.diff(data['layers'])
        ax2.plot(changes, color='magenta', marker='o')
        ax2.set_title("Layer/Size Changes")
        ax2.set_xlabel("Index")
        ax2.set_ylabel("Change in Layer/Size")
        ax2.grid(True)

        fig.tight_layout()
        self.canvases['layers'].draw()

    # ---------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------

    def _show_no_data(self, fig: Figure):
        """
        Display a "No Data" text on the figure if there's nothing to plot.
        """
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5, "No data available",
            ha='center', va='center', fontsize=14
        )
        ax.set_axis_off()
        fig.tight_layout()

    def refresh(self):
        """
        Refresh all sub-views. If auto_refresh is True, or user clicked Refresh,
        we re-draw everything.
        """
        start = time.time()
        self._update_status("Refreshing views...")

        self._update_combined_view()
        self._update_entropy_view()
        self._update_timeline_view()
        self._update_layer_view()

        end = time.time()
        self._update_status(f"Refreshed in {end - start:.2f}s")

    def _on_view_changed(self, event):
        """
        Called when user changes the selection in the 'View' combobox.
        We switch the Notebook tab to match, then refresh if auto_refresh is True.
        """
        self.current_view = self.view_var.get()
        tab_index = ["combined", "entropy", "timeline", "layers"].index(self.current_view)
        self.viz_notebook.select(tab_index)

        if self.auto_refresh.get():
            self.refresh()

    def _save_current_plot(self):
        """
        Save the figure corresponding to the current view to a file.
        """
        view = self.current_view
        if view not in self.figures:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"quantum_stack_{view}_visual"
        )
        if filename:
            try:
                self.figures[view].savefig(filename, dpi=300, bbox_inches="tight")
                self._update_status(f"Plot saved to {filename}")
            except Exception as e:
                self._update_status(f"Error saving plot: {str(e)}")

    def _reset_view(self):
        """
        Clear the figure for the current view, then (optionally) re-draw if auto_refresh is True.
        """
        view = self.current_view
        if view in self.figures:
            self.figures[view].clear()
            self.canvases[view].draw()

        if self.auto_refresh.get():
            self.refresh()

    def _update_status(self, msg: str):
        """Update the status label with a given message."""
        self.status_var.set(msg)
        self.update()

    def clear(self):
        """
        Clears all figure content from each sub-view, effectively resetting the tab.
        """
        for fig in self.figures.values():
            fig.clear()
        for canvas in self.canvases.values():
            canvas.draw()
        self._update_status("Visualizations cleared.")
