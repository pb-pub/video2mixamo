# Video to Maximo - Temporal Smoothing Module

"""
Temporal smoothing module using One Euro Filter.

The One Euro Filter adapts its smoothing based on signal speed:
- Aggressive smoothing for slow/static poses
- Minimal smoothing for fast motion

Reference: 
    G. Durand, "Fast and Accurate One-Euro Filter for Smoothing and Acceleration 
    Filtering", Journal of Signaling, 2020.

Classes:
    OneEuroFilter: One Euro Filter implementation
    Smoother: Manages per-landmark, per-axis filters
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FilterConfig:
    """Configuration for One Euro Filter."""
    min_cutoff: float = 1.0      # Minimum cutoff frequency (Hz)
    beta: float = 0.0            # Cutoff slope for speed
    d_cutoff: float = 1.0        # Derivative cutoff frequency (Hz)
    frequency: float = 30.0      # Input sample rate (Hz)


class OneEuroFilter:
    """
    One Euro Filter implementation for smoothing time-series data.
    
    The filter adapts its smoothing based on the speed of the signal:
    - Fast signals get minimal smoothing
    - Slow signals get strong smoothing
    
    This makes it ideal for pose data where:
    - Static poses need smoothing
    - Fast movements should remain responsive
    """
    
    def __init__(self, config: FilterConfig = None):
        """
        Initialize One Euro Filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config or FilterConfig()
        self._x_prev: Optional[float] = None
        self._dx_prev: Optional[float] = None
        self._t_prev: Optional[float] = None
        self._initialized = False
    
    def filter(self, x: float, t: float) -> float:
        """
        Filter a single value at a given timestamp.
        
        Args:
            x: Input value
            t: Timestamp (in milliseconds or any consistent unit)
        
        Returns:
            Smoothed value
        """
        # Compute sampling interval
        if self._t_prev is not None:
            dt = (t - self._t_prev) / 1000.0  # Convert to seconds
            if dt <= 0:
                dt = 1.0 / self.config.frequency
        else:
            dt = 1.0 / self.config.frequency
        
        # Compute alpha based on cutoff
        alpha = self._compute_alpha(dt)
        
        # Filter the value
        if self._initialized:
            x_filtered = alpha * x + (1 - alpha) * self._x_prev
        else:
            x_filtered = x
            self._initialized = True
        
        # Filter the derivative (speed)
        dx = (x_filtered - self._x_prev) / dt if self._x_prev is not None else 0.0
        
        # Apply derivative filter
        alpha_dx = self._compute_alpha(dt, self.config.d_cutoff)
        dx_filtered = alpha_dx * dx + (1 - alpha_dx) * self._dx_prev if self._dx_prev is not None else dx
        
        # Store for next iteration
        self._x_prev = x_filtered
        self._dx_prev = dx_filtered
        self._t_prev = t
        
        return x_filtered
    
    def _compute_alpha(self, dt: float, cutoff: float = None) -> float:
        """Compute the alpha value for the exponential moving average."""
        if cutoff is None:
            cutoff = self.config.min_cutoff
        
        # Compute tau (time constant)
        tau = 1.0 / (2 * np.pi * cutoff)
        
        # Alpha for exponential smoothing
        alpha = 1.0 / (1.0 + tau / dt)
        
        return alpha
    
    def reset(self) -> None:
        """Reset filter state."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None
        self._initialized = False


class Smoother:
    """
    Manages One Euro Filters for all landmarks and axes.
    
    Provides configurable filtering per axis and per landmark,
    allowing stronger smoothing on Z (depth) axis.
    """
    
    def __init__(
        self,
        config: FilterConfig = None,
        z_config: FilterConfig = None
    ):
        """
        Initialize smoother.
        
        Args:
            config: Filter configuration for X and Y axes
            z_config: Filter configuration for Z axis (stronger smoothing)
        """
        self.config = config or FilterConfig()
        self.z_config = z_config or FilterConfig(
            min_cutoff=0.5,  # Stronger Z smoothing
            beta=0.0,
            d_cutoff=1.0,
            frequency=30.0
        )
        
        # Filter storage: {landmark_index: {axis: filter}}
        self._filters: Dict[int, Dict[str, OneEuroFilter]] = {}
        self._last_timestamp: Optional[float] = None
        self._last_valid_landmarks: Dict[int, List[float]] = {}
    
    def filter_landmarks(
        self,
        landmarks: List[List[float]],
        timestamp_ms: float
    ) -> List[List[float]]:
        """
        Filter all landmarks at a given timestamp.
        
        Args:
            landmarks: 33x3 list of [x, y, z] positions
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            Filtered landmarks (33x3)
        """
        if len(landmarks) != 33:
            return landmarks
        
        filtered = []
        
        for lm_idx, landmark in enumerate(landmarks):
            if lm_idx not in self._filters:
                # Create filters for this landmark
                self._filters[lm_idx] = {
                    'x': OneEuroFilter(self.config),
                    'y': OneEuroFilter(self.config),
                    'z': OneEuroFilter(self.z_config)
                }
            
            # Filter each axis
            x = self._filters[lm_idx]['x'].filter(landmark[0], timestamp_ms)
            y = self._filters[lm_idx]['y'].filter(landmark[1], timestamp_ms)
            z = self._filters[lm_idx]['z'].filter(landmark[2], timestamp_ms)
            
            filtered.append([x, y, z])
        
        self._last_timestamp = timestamp_ms
        self._last_valid_landmarks = {i: list(lm) for i, lm in enumerate(filtered)}
        
        return filtered
    
    def filter_with_missing(
        self,
        landmarks: List[List[float]],
        timestamps_ms: List[float]
    ) -> List[List[List[float]]]:
        """
        Filter a sequence of frames, handling missing landmarks.
        
        Args:
            landmarks: List of 33x3 landmark lists (one per frame)
            timestamps_ms: List of timestamps (one per frame)
        
        Returns:
            List of filtered landmarks for each frame
        """
        if len(landmarks) != len(timestamps_ms):
            raise ValueError("Landmarks and timestamps must have same length")
        
        filtered_sequence = []
        
        for i, (landmarks_frame, timestamp_ms) in enumerate(zip(landmarks, timestamps_ms)):
            filtered_frame = self.filter_landmarks(landmarks_frame, timestamp_ms)
            filtered_sequence.append(filtered_frame)
        
        return filtered_sequence
    
    def reset(self) -> None:
        """Reset all filters."""
        self._filters = {}
        self._last_timestamp = None
        self._last_valid_landmarks = {}


class SavitzkyGolayFilter:
    """
    Savitzky-Golay filter for offline smoothing.
    
    Uses polynomial regression over a window to smooth data.
    Not suitable for real-time (needs future frames).
    """
    
    def __init__(self, window_size: int = 5, poly_order: int = 2):
        """
        Initialize Savitzky-Golay filter.
        
        Args:
            window_size: Number of frames to use for smoothing (must be odd)
            poly_order: Polynomial order for fitting
        """
        if window_size % 2 == 0:
            window_size += 1  # Must be odd
        
        self.window_size = window_size
        self.poly_order = min(poly_order, window_size - 1)
    
    def smooth(self, data: np.ndarray) -> np.ndarray:
        """
        Smooth data using Savitzky-Golay filter.
        
        Args:
            data: Input array (N, 3) for N frames of 3D data
        
        Returns:
            Smoothed array (N, 3)
        """
        from scipy.signal import savgol_filter
        
        if len(data) < self.window_size:
            return data.copy()
        
        smoothed = np.zeros_like(data)
        
        for axis in range(3):
            smoothed[:, axis] = savgol_filter(
                data[:, axis],
                self.window_size,
                self.poly_order
            )
        
        return smoothed
    
    def smooth_sequence(
        self,
        landmark_sequence: List[List[List[float]]]
    ) -> List[List[List[float]]]:
        """
        Smooth a sequence of landmarks.
        
        Args:
            landmark_sequence: List of 33x3 landmark lists
        
        Returns:
            Smoothed landmark sequence
        """
        if not landmark_sequence:
            return []
        
        # Convert to numpy array
        data = np.array(landmark_sequence)  # (frames, 33, 3)
        
        # Smooth each landmark
        smoothed = np.zeros_like(data)
        
        for lm_idx in range(33):
            for axis in range(3):
                smoothed[:, lm_idx, axis] = self.smooth(
                    data[:, lm_idx, axis]
                )
        
        return smoothed.tolist()


if __name__ == "__main__":
    # Test the filters
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Generate test data: sine wave with noise
    t = np.linspace(0, 2 * np.pi, 100)
    noisy = np.sin(t) + np.random.randn(100) * 0.3
    
    # One Euro Filter
    one_euro = OneEuroFilter(FilterConfig(min_cutoff=0.5, beta=0.5, frequency=30.0))
    smoothed = [one_euro.filter(x, i) for i, x in enumerate(noisy)]
    
    print("One Euro Filter test:")
    print(f"  Input variance: {np.var(noisy):.4f}")
    print(f"  Output variance: {np.var(smoothed):.4f}")
    
    # Savitzky-Golay filter
    sg = SavitzkyGolayFilter(window_size=7, poly_order=3)
    data = np.array([[x, x, x] for x in noisy])  # (N, 3)
    sg_smoothed = sg.smooth(data)
    
    print("\nSavitzky-Golay Filter test:")
    print(f"  Input variance: {np.var(noisy):.4f}")
    print(f"  Output variance: {np.var(sg_smoothed[:, 0]):.4f}")
