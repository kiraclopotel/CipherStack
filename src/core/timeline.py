# src/core/timeline.py

from typing import Dict, List, Tuple, Optional
import time
import hashlib
import numpy as np
from .mathematics import MathematicalCore

class TimelineManager:
    """
    Manages timeline markers for each seed/message.
    Integrates with MathematicalCore to compute layers and maintain
    consistent 'timeline continuity' for stacked encryption logic.
    """

    def __init__(self):
        # each marker is { 'seed', 'id', 'timestamp', 'entropy', 'layer', ... }
        self.markers: Dict[int, Dict] = {}
        self.checksum_history: List[str] = []

        # This calls your refined mathematics.py
        self.mathematical_core = MathematicalCore()

        # For statistics
        self.layer_history: List[int] = []
        self.timeline_entries: Dict[int, List[Tuple[int, int]]] = {}
        self.timeline_stats: Dict[str, float] = {'total_entries': 0, 'max_depth': 0}

        # For continuity checks
        self.previous_timeline: Optional[List[Tuple[int,int]]] = None
        self.last_timestamp = 0.0

    def create_marker(self, seed: int, message_id: int, message: bytes, entropy: float) -> Dict:
        """
        Create a timeline marker with advanced mathematical properties.
        1) Generate a timeline from 'seed'.
        2) Possibly ensure continuity with previous timeline.
        3) Compute layer from seed, manage layer states, etc.
        4) Create a monotonic timestamp.
        5) Store stats and generate a marker with an enhanced checksum.
        """
        # 1) Generate timeline
        timeline = self.mathematical_core.generate_entry_timeline(seed)

        # 2) Possibly ensure continuity with previous
        if self.previous_timeline:
            timeline = self._ensure_timeline_continuity(timeline)

        # 3) Compute and validate layer
        layer = self.mathematical_core.compute_layer(seed)
        layer = self.mathematical_core._manage_layer_state(seed, layer)

        # 4) Monotonic timestamp
        current_time = time.time()
        if current_time <= self.last_timestamp:
            current_time = self.last_timestamp + 0.001
        self.last_timestamp = current_time

        # 5) Update statistics
        self._update_statistics(timeline)

        # Save timeline, layer, etc.
        self.timeline_entries[seed] = timeline
        self.layer_history.append(layer)
        self.previous_timeline = timeline

        # Build final marker dict
        marker = {
            'seed': seed,
            'id': message_id,
            'timestamp': current_time,
            'entropy': entropy,
            'layer': layer,
            'timeline': timeline,
            'depth': len(timeline),
            'checksum': self._generate_checksum(seed, message_id, message, timeline)
        }
        self.markers[seed] = marker
        return marker

    def _generate_checksum(self, seed: int, message_id: int,
                           message: bytes, timeline: List[Tuple[int,int]]) -> str:
        """
        Generate an enhanced checksum that includes:
          - seed
          - message_id
          - raw message bytes
          - timeline representation
          - layer_function result (to add variability)
          - current time
        Then store that checksum in the local history.
        """
        timeline_str = str(timeline)
        layer_val = self.mathematical_core.compute_layer(seed)
        layer_output = str(self.mathematical_core.layer_function(seed, layer_val))

        # Combine for the checksum
        components = [
            str(seed).encode(),
            str(message_id).encode(),
            message,
            timeline_str.encode(),
            layer_output.encode(),
            str(time.time()).encode()
        ]
        checksum_data = b''.join(components)
        checksum = hashlib.sha256(checksum_data).hexdigest()
        self.checksum_history.append(checksum)
        return checksum

    def _ensure_timeline_continuity(self, new_timeline: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
        """
        Ensure smooth transition between self.previous_timeline and new_timeline
        by adjusting the 'depth' if there's a large jump.
        """
        if not self.previous_timeline:
            return new_timeline

        last_depth = self.previous_timeline[-1][1]
        first_depth = new_timeline[0][1]

        if abs(last_depth - first_depth) > 1:
            # shift entire new timeline so it lines up
            depth_diff = last_depth - first_depth
            adjusted = [(digit, depth + depth_diff) for (digit, depth) in new_timeline]
            return adjusted
        return new_timeline

    def verify_marker(self, seed: int, message_id: int, message: bytes) -> bool:
        """
        Verify the integrity of a marker:
          1) Check if it exists
          2) Confirm continuity with previous timeline
          3) Re-generate checksum and compare
          4) Confirm layer matches the stored layer
        """
        if seed not in self.markers:
            return False
        marker = self.markers[seed]
        timeline = marker['timeline']

        # continuity check if possible
        if message_id > 0 and (seed - 1) in self.markers:
            prev_timeline = self.markers[seed - 1]['timeline']
            if not self._verify_timeline_continuity(prev_timeline, timeline):
                return False

        # verify checksum
        current_checksum = self._generate_checksum(seed, message_id, message, timeline)
        if marker['checksum'] != current_checksum:
            return False

        # verify layer
        current_layer = self.mathematical_core.compute_layer(seed)
        stored_layer = marker['layer']
        if current_layer != stored_layer:
            return False

        return True

    def _verify_timeline_continuity(self,
                                    prev_timeline: List[Tuple[int,int]],
                                    curr_timeline: List[Tuple[int,int]]
    ) -> bool:
        """Check if the end of prev_timeline aligns smoothly with start of curr_timeline."""
        if not prev_timeline or not curr_timeline:
            return True
        last_depth = prev_timeline[-1][1]
        first_depth = curr_timeline[0][1]
        return abs(last_depth - first_depth) <= 1

    def _update_statistics(self, timeline: List[Tuple[int,int]]) -> None:
        """
        Update local timeline stats:
          - increment total_entries
          - compare max_depth
        """
        self.timeline_stats['total_entries'] += 1
        depths = [depth for (_digit, depth) in timeline]
        if depths:
            max_depth = max(depths)
            if max_depth > self.timeline_stats['max_depth']:
                self.timeline_stats['max_depth'] = max_depth

    def get_layer_statistics(self) -> Dict[str, float]:
        """
        Compute basic layer stats from self.markers.
        Return dict with mean_layer, std_layer, min_layer, max_layer, total_layers
        """
        layers = [m['layer'] for m in self.markers.values()]
        if not layers:
            return {
                'mean_layer': 0.0,
                'std_layer': 0.0,
                'min_layer': 0.0,
                'max_layer': 0.0,
                'total_layers': 0
            }

        mean_lyr = float(np.mean(layers))
        std_lyr = float(np.std(layers))
        min_lyr = int(np.min(layers))
        max_lyr = int(np.max(layers))
        unique_layers = len(set(layers))

        return {
            'mean_layer': mean_lyr,
            'std_layer': std_lyr,
            'min_layer': min_lyr,
            'max_layer': max_lyr,
            'total_layers': unique_layers
        }

    def get_timeline_metrics(self, seed: int) -> Optional[Dict]:
        """
        Retrieve metrics for timeline of a given seed:
         - max_depth, min_depth, avg_depth
         - layer, timestamp, entropy, etc.
        """
        if seed not in self.markers:
            return None

        marker = self.markers[seed]
        timeline = marker['timeline']

        depths = [d for (_digit, d) in timeline]
        return {
            'max_depth': max(depths) if depths else 0,
            'min_depth': min(depths) if depths else 0,
            'avg_depth': float(np.mean(depths)) if depths else 0.0,
            'layer': marker['layer'],
            'timestamp': marker['timestamp'],
            'entropy': marker['entropy'],
            'timeline_length': len(timeline),
            'checksum_count': len(self.checksum_history)
        }

    def get_visualization_data(self) -> Dict:
        """
        Prepare data for visualization: timestamps, layers, entropies, etc.
        """
        if not self.markers:
            return {
                'timestamps': [],
                'layers': [],
                'entropies': [],
                'depths': [],
                'seeds': [],
                'checksum_counts': []
            }

        data = {
            'timestamps': [],
            'layers': [],
            'entropies': [],
            'depths': [],
            'seeds': [],
            'checksum_counts': []
        }

        for seed, marker in self.markers.items():
            data['timestamps'].append(marker['timestamp'])
            data['layers'].append(marker['layer'])
            data['entropies'].append(marker['entropy'])
            data['depths'].append(len(marker['timeline']))
            data['seeds'].append(seed)
            # How many times does marker['checksum'] appear in self.checksum_history
            c_count = self.checksum_history.count(marker['checksum'])
            data['checksum_counts'].append(c_count)

        return data

    # --------------------------------------------------------------------------
    # Minimal restore_from_data for encryption.load_state() usage
    # --------------------------------------------------------------------------
    def restore_from_data(self, timeline_data: Dict[str, List[float]]):
        """
        Called by encryption.load_state() if timeline_data is present.
        We rebuild minimal markers from seeds/timestamps/layers/entropies.
        Expand further if you want more thorough restoration.
        """
        try:
            if not timeline_data or not isinstance(timeline_data, dict):
                return

            seeds = timeline_data.get('seeds', [])
            timestamps = timeline_data.get('timestamps', [])
            entropies = timeline_data.get('entropies', [])
            layers = timeline_data.get('layers', [])
            depth_counts = timeline_data.get('depths', [])
            checksums = timeline_data.get('checksum_counts', [])

            for i, seed_val in enumerate(seeds):
                # We do minimal logic here:
                #   generate a "timeline" as empty or mock
                #   parse time, layer, etc. 
                t_val = timestamps[i] if i < len(timestamps) else time.time()
                e_val = entropies[i] if i < len(entropies) else 1.0
                l_val = layers[i] if i < len(layers) else 1
                c_count = checksums[i] if i < len(checksums) else 1

                # We'll store an empty timeline by default, or you can store a mock
                fake_timeline = [(l_val, l_val)] * (depth_counts[i] if i < len(depth_counts) else 0)

                marker = {
                    'seed': seed_val,
                    'id': i,
                    'timestamp': t_val,
                    'entropy': e_val,
                    'layer': l_val,
                    'timeline': fake_timeline,
                    'depth': len(fake_timeline),
                    'checksum': "restored_checksum"  # placeholder
                }
                self.markers[seed_val] = marker

            # If we want to re-create self.checksum_history up to the count indicated:
            # e.g. sum of all checksums => not always needed
            # We'll just do a minimal approach:
            for i in range(len(seeds)):
                self.checksum_history.append("restored_checksum")

        except Exception as e:
            print(f"Error restoring timeline data in TimelineManager: {e}")
            self.markers.clear()
            self.checksum_history.clear()

    def reset(self) -> None:
        """
        Clear all timeline state, marker info,
        and reset the MathematicalCore as well.
        """
        self.markers.clear()
        self.checksum_history.clear()
        self.layer_history.clear()
        self.timeline_entries.clear()
        self.timeline_stats = {'total_entries': 0, 'max_depth': 0}
        self.previous_timeline = None
        self.last_timestamp = 0
        self.mathematical_core.clear_caches()
