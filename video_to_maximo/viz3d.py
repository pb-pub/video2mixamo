"""
Interactive 3D skeleton visualizer for MediaPipe pose landmarks.

Opens a separate OpenGL window showing all 33 pose landmarks and
skeleton connections in real time. The window can be freely rotated,
panned, and zoomed without resetting between frames.

Requires: pyqtgraph >= 0.13, PyOpenGL, PyQt5 (or PySide2/PySide6)
    pip install pyqtgraph PyOpenGL PyQt5

Usage (from main.py):
    viz = Pose3DVisualizer()
    viz.open()
    # each frame:
    viz.update_landmarks(result.pose_world_landmarks, visibility)
    # to close:
    viz.close()
"""

import threading
import queue
from typing import List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe 33-landmark skeleton connections (index pairs)
# ---------------------------------------------------------------------------
_CONNECTIONS: List[Tuple[int, int]] = [
    # Face outline
    (0, 1), (1, 2), (2, 3), (3, 7),    # nose → left eye → ear
    (0, 4), (4, 5), (5, 6), (6, 8),    # nose → right eye → ear
    (9, 10),                             # mouth corners
    # Torso
    (11, 12),                            # shoulders
    (11, 23), (12, 24),                  # shoulder → hip
    (23, 24),                            # hips
    # Left arm
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),        # wrist → fingers
    (17, 19),
    # Right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),        # wrist → fingers
    (18, 20),
    # Left leg
    (23, 25), (25, 27),
    (27, 29), (29, 31), (27, 31),        # ankle → heel/foot
    # Right leg
    (24, 26), (26, 28),
    (28, 30), (30, 32), (28, 32),        # ankle → heel/foot
]

# Indices considered part of the right side (red tint)
_RIGHT_INDICES = {4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
# Indices considered part of the left side (blue tint)
_LEFT_INDICES  = {1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}


def _joint_colors(n: int = 33) -> np.ndarray:
    """Build per-joint RGBA color array (left=cyan, right=red, center=white)."""
    colors = np.ones((n, 4), dtype=np.float32)
    colors[:, :] = (1.0, 1.0, 1.0, 1.0)           # default white
    for i in _LEFT_INDICES:
        if i < n:
            colors[i] = (0.2, 0.8, 1.0, 1.0)       # cyan
    for i in _RIGHT_INDICES:
        if i < n:
            colors[i] = (1.0, 0.35, 0.35, 1.0)     # red
    return colors


class Pose3DVisualizer:
    """
    Real-time interactive 3D skeleton window.

    Runs in a daemon thread with its own Qt event loop so it never
    blocks the main OpenCV capture loop.  The camera view (rotation,
    zoom, pan) is preserved between data updates.

    Parameters
    ----------
    window_size : tuple[int, int]
        Initial width and height of the window in pixels.
    """

    def __init__(self, window_size: Tuple[int, int] = (700, 700)):
        self._window_size = window_size
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while the visualizer window is open."""
        return self._thread is not None and self._thread.is_alive()

    def open(self) -> None:
        """Open (or re-open) the 3D visualizer window."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_qt, daemon=True, name="viz3d"
        )
        self._thread.start()

    def close(self) -> None:
        """Signal the window to close."""
        self._stop_event.set()

    def toggle(self) -> bool:
        """Toggle open/closed. Returns True if now open."""
        if self.is_running:
            self.close()
            return False
        self.open()
        return True

    def update_landmarks(
        self,
        world_landmarks: List[List[float]],
        visibility: Optional[List[float]] = None,
    ) -> None:
        """
        Push the latest landmarks for display.

        Parameters
        ----------
        world_landmarks : list of [x, y, z]
            33 MediaPipe world-space pose landmarks (meters, Y-up).
        visibility : list of float, optional
            Per-landmark visibility scores [0, 1].
        """
        if not self.is_running or world_landmarks is None:
            return
        # Keep only the most recent frame — drain the queue first
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait((world_landmarks, visibility))
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Qt event loop (daemon thread)
    # ------------------------------------------------------------------

    def _run_qt(self) -> None:
        try:
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl
            from pyqtgraph.Qt import QtWidgets, QtCore
        except ImportError:
            print(
                "\n[Viz3D] Required packages not installed.\n"
                "  Install with:\n"
                "    pip install pyqtgraph PyOpenGL PyQt5\n"
            )
            return

        try:
            pg.setConfigOption("antialias", True)

            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication([])

            w = gl.GLViewWidget()
            w.setWindowTitle("3D Pose Landmarks  |  drag=rotate  scroll=zoom  ctrl+drag=pan")
            w.resize(*self._window_size)
            w.show()
            # Initial camera: looking slightly from above and to the side
            w.setCameraPosition(distance=2.5, elevation=20, azimuth=-60)

            # ---- static scene items ----
            grid = gl.GLGridItem()
            grid.setSize(2, 2)
            grid.setSpacing(0.1, 0.1)
            # place grid roughly at feet level (Y ~ -1 in Y-up coords)
            grid.translate(0, 0, -1.0)
            w.addItem(grid)

            axis = gl.GLAxisItem()
            axis.setSize(0.25, 0.25, 0.25)
            w.addItem(axis)

            # ---- joints ----
            _colors = _joint_colors(33)
            scatter = gl.GLScatterPlotItem(
                pos=np.zeros((33, 3), dtype=np.float32),
                color=_colors,
                size=10,
                pxMode=True,
            )
            w.addItem(scatter)

            # ---- bones: one GLLinePlotItem per connection for compatibility ----
            bone_items: List[gl.GLLinePlotItem] = []
            for _ in _CONNECTIONS:
                item = gl.GLLinePlotItem(
                    pos=np.zeros((2, 3), dtype=np.float32),
                    color=(0.9, 0.65, 0.1, 0.85),
                    width=2.0,
                    antialias=True,
                )
                w.addItem(item)
                bone_items.append(item)

            # ---- text overlay (basic info) ----
            # pyqtgraph GLViewWidget doesn't have built-in text; skip for simplicity

            # ---- timer-driven update ----
            def _tick() -> None:
                if self._stop_event.is_set():
                    _timer.stop()
                    w.close()
                    app.quit()
                    return

                try:
                    data = self._queue.get_nowait()
                except queue.Empty:
                    return

                world_lms, vis = data
                if not world_lms or len(world_lms) != 33:
                    return

                # Convert to numpy.
                # pyqtgraph GLViewWidget: grid lies in XY plane → Z is the UP axis.
                # MediaPipe world coords: X=left/right, Y=up, Z=depth (neg = toward cam).
                # Mapping:
                #   display X = -mediapipe X         (negate to fix mirror)
                #   display Y = -mediapipe Z         (depth; negate so face looks toward +Y)
                #   display Z = -mediapipe Y         (MediaPipe Y is down, negate → Z-up)
                pts = np.array(
                    [[-lm[0], -lm[2], -lm[1]] for lm in world_lms],
                    dtype=np.float32,
                )

                # Update joint colors by visibility
                colors = _joint_colors(33).copy()
                if vis and len(vis) == 33:
                    for i, v in enumerate(vis):
                        colors[i, 3] = max(0.15, float(v))   # alpha = visibility
                scatter.setData(pos=pts, color=colors)

                # Update bones
                for idx, (a, b) in enumerate(_CONNECTIONS):
                    if a < len(pts) and b < len(pts):
                        bone_items[idx].setData(
                            pos=np.array([pts[a], pts[b]], dtype=np.float32)
                        )

            _timer = QtCore.QTimer()
            _timer.timeout.connect(_tick)
            _timer.start(33)   # ~30 fps polling

            app.exec_()

        except Exception as exc:
            print(f"[Viz3D] Unexpected error: {exc}")
