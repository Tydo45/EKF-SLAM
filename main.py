import sys
import math
import argparse
import json
import numpy as np
from matplotlib.patches import Ellipse

#!/usr/bin/env python3
"""
Simple EKF-SLAM visualizer entrypoint.

Run this file to open a small interactive visualizer window that:
- draws a robot pose (arrow),
- draws landmarks (points + IDs),
- draws a 2D covariance ellipse for the robot pose,
- allows stepping a simple simulated motion with space, or runs continuously with 'r',
- press 'q' to quit.

This is a self-contained starter visualizer for integration into an EKF-SLAM project.
"""

import matplotlib.pyplot as plt

# --- Utilities ---------------------------------------------------------------

def cov_ellipse_params(cov, n_std=2.0):
    """Return width, height, angle (degrees) for an ellipse representing cov."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort largest first
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 0.0))
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle

# --- Visualizer class -------------------------------------------------------

class Visualizer:
    def __init__(self, landmarks=None, robot_pose=None, robot_cov=None):
        # landmarks: list of (id, x, y)
        self.landmarks = landmarks or []
        # robot_pose: (x, y, theta)
        self.pose = robot_pose or (0.0, 0.0, 0.0)
        # robot covariance 2x2 (x,y)
        self.cov = robot_cov if robot_cov is not None else np.array([[0.2, 0.0], [0.0, 0.2]])
        # matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.grid(True)
        self.running = False
        self._init_draw()
        # connect events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _init_draw(self):
        self.landmark_sc = self.ax.scatter([], [], c='tab:green', s=50, zorder=2)
        self.landmark_texts = []
        self.robot_arrow = None
        self.cov_ellipse = None
        self._update_plot_limits()

    def _update_plot_limits(self):
        # adjust limits based on landmarks and robot
        xs = [x for (_id, x, y) in self.landmarks] + [self.pose[0]]
        ys = [y for (_id, x, y) in self.landmarks] + [self.pose[1]]
        if xs and ys:
            xmin, xmax = min(xs) - 5, max(xs) + 5
            ymin, ymax = min(ys) - 5, max(ys) + 5
        else:
            xmin, xmax, ymin, ymax = -10, 10, -10, 10
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

    def _draw_landmarks(self):
        if self.landmark_texts:
            for t in self.landmark_texts:
                t.remove()
            self.landmark_texts = []
        if self.landmarks:
            xs = [x for (_id, x, y) in self.landmarks]
            ys = [y for (_id, x, y) in self.landmarks]
            self.landmark_sc.set_offsets(np.column_stack((xs, ys)))
            for (_id, x, y) in self.landmarks:
                txt = self.ax.text(x, y, f'{_id}', color='k', fontsize=9,
                                   verticalalignment='bottom', horizontalalignment='right')
                self.landmark_texts.append(txt)
        else:
            self.landmark_sc.set_offsets([])

    def _draw_robot(self):
        x, y, theta = self.pose
        # remove previous graphics
        if self.robot_arrow is not None:
            self.robot_arrow.remove()
            self.robot_arrow = None
        if self.cov_ellipse is not None:
            self.cov_ellipse.remove()
            self.cov_ellipse = None
        # arrow indicating heading
        dx = math.cos(theta) * 1.0
        dy = math.sin(theta) * 1.0
        self.robot_arrow = self.ax.arrow(x, y, dx, dy, head_width=0.4, head_length=0.6,
                                         fc='tab:blue', ec='tab:blue', zorder=3)
        # covariance ellipse (use xy submatrix)
        width, height, angle = cov_ellipse_params(self.cov)
        self.cov_ellipse = Ellipse((x, y), width, height, angle=angle,
                                   edgecolor='tab:blue', facecolor='tab:blue', alpha=0.15, zorder=1)
        self.ax.add_patch(self.cov_ellipse)

    def update(self):
        """Redraw everything (call after updating self.pose/self.landmarks/self.cov)."""
        self._update_plot_limits()
        self._draw_landmarks()
        self._draw_robot()
        self.fig.canvas.draw_idle()

    # --- simple simulation step (replaceable by EKF update calls) ------------
    def step_simulation(self, v=0.5, w=0.1, dt=0.5):
        """Move robot forward with differential motion and grow cov slightly."""
        x, y, theta = self.pose
        # simple bicycle/differential motion integration
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        self.pose = (x, y, theta)
        # grow covariance as a simple model
        Q = np.array([[0.02 * abs(v) + 0.001, 0.0], [0.0, 0.02 * abs(v) + 0.001]])
        self.cov = self.cov + Q
        # optionally add a landmark occasionally
        if np.random.rand() < 0.05:
            new_id = (max([_id for (_id, _, _) in self.landmarks], default=0) + 1)
            lx = x + np.random.randn() * 2 + 5 * math.cos(theta + np.random.randn())
            ly = y + np.random.randn() * 2 + 5 * math.sin(theta + np.random.randn())
            self.landmarks.append((new_id, lx, ly))

    # --- event handling -------------------------------------------------------
    def _on_key(self, event):
        if event.key == ' ':
            self.step_simulation()
            self.update()
        elif event.key == 'r':
            self.running = not self.running
            print("Running:", self.running)
        elif event.key == 'q':
            plt.close(self.fig)

    def run(self):
        """Start visualizer main loop. Press space to step, 'r' to run continuously, 'q' to quit."""
        self.update()
        # simple loop to allow toggled running
        try:
            while plt.fignum_exists(self.fig.number):
                if self.running:
                    self.step_simulation()
                    self.update()
                    plt.pause(0.1)
                else:
                    plt.pause(0.05)
        except KeyboardInterrupt:
            pass

# --- Data loader -------------------------------------------------------------
class DataLoader:
    """Simple JSONL data loader for the visualizer.

    Usage:
      dl = DataLoader(path, start_frame=None)
      frame = dl.read_next()           # read next frame (returns (landmarks, pose, cov))
      frame = dl.read_frame(10)        # read frame index 10 (0-based)
      total = dl.count_frames()

    The loader is robust to plain JSON (single object) or newline-delimited
    JSON objects (JSONL). For JSONL it treats each non-empty line as a frame.
    """
    def __init__(self, file_path, start_frame=None):
        self.file_path = file_path
        # current frame index last returned by read_next/read_frame (-1 = none yet)
        self.current_index = -1
        # optional starting frame if you want to begin at a particular index
        self.start_frame = start_frame

    def _parse_record(self, data):
        """Convert a parsed JSON dict into (landmarks, pose, cov) like load_from_json."""
        pose = None
        landmarks = []
        cov = None

        if not isinstance(data, dict):
            raise ValueError("Expected JSON object for frame")

        if 'robot_position' in data:
            rp = data['robot_position']
            pose = (float(rp[0]), float(rp[1]), float(rp[2]))
        elif 'pose' in data: # should not be the case
            p = data['pose']
            if isinstance(p, (list, tuple)):
                pose = (float(p[0]), float(p[1]), float(p[2]))
            elif isinstance(p, dict):
                pose = (float(p.get('x', 0.0)), float(p.get('y', 0.0)), float(p.get('theta', 0.0)))

        # landmarks / map
        if 'map' in data and isinstance(data['map'], (list, tuple)):
            mp = data['map']
            for i, item in enumerate(mp):
                try:
                    x, y = float(item[0]), float(item[1])
                except Exception:
                    continue
                landmarks.append((i + 1, x, y))
        elif 'landmarks' in data: # should not be the case, we use map, not landmarks
            lm = data['landmarks']
            for i, item in enumerate(lm):
                if isinstance(item, dict):
                    _id = item.get('id', i + 1)
                    x = item.get('x') if 'x' in item else (item.get('0') if '0' in item else None)
                    y = item.get('y') if 'y' in item else (item.get('1') if '1' in item else None)
                    try:
                        landmarks.append((int(_id), float(x), float(y)))
                    except Exception:
                        continue
                elif isinstance(item, (list, tuple)):
                    if len(item) == 3:
                        try:
                            landmarks.append((int(item[0]), float(item[1]), float(item[2])))
                        except Exception:
                            continue
                    elif len(item) == 2:
                        try:
                            landmarks.append((i + 1, float(item[0]), float(item[1])))
                        except Exception:
                            continue

        # covariance - just the shape of the robot circle, TODO change this later
        if 'cov' in data:
            try:
                arr = np.array(data['cov'], dtype=float)
                if arr.shape == (2, 2):
                    cov = arr
            except Exception:
                cov = None

        if pose is None:
            pose = (0.0, 0.0, 0.0)
        if cov is None:
            cov = np.array([[0.5, 0.0], [0.0, 0.3]])

        return landmarks, pose, cov

    def _iter_nonempty_lines(self):
        """Yield (idx, line) for each non-empty line in the file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, ln in enumerate(f):
                if ln.strip():
                    yield i, ln.rstrip('\n')

    def count_frames(self):
        """Return number of non-empty JSONL frames (or 1 for plain JSON)."""
        # count non-empty lines
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if not text:
                return 0
            # if file parses as a single JSON object (multi-line), treat as one frame
            try:
                _ = json.loads(text)
                # if it's a dict or list and not line-delimited, count as 1
                return 1
            except Exception:
                # not a single JSON object, count non-empty lines
                for ln in text.splitlines():
                    if ln.strip():
                        count += 1
        return count

    def read_frame(self, index):
        """Read a specific frame (0-based index) and return (landmarks, pose, cov).

        Raises IndexError when index is out of range.
        """
        if index is None:
            index = 0
        # iterate until the requested frame
        for i, ln in self._iter_nonempty_lines():
            if i == index:
                try:
                    data = json.loads(ln)
                except Exception:
                    # fallback: try to parse whole file as JSON and treat as single frame
                    with open(self.file_path, 'r', encoding='utf-8') as f:
                        full = f.read()
                    data = json.loads(full)
                self.current_index = index
                return self._parse_record(data)
        raise IndexError(f"Frame index {index} out of range")

    def read_next(self):
        """Read the next frame sequentially and return (landmarks, pose, cov).

        If called repeatedly, this will traverse frames from 0..N-1. After the
        last frame, raises StopIteration.
        """
        next_index = self.current_index + 1
        try:
            return self.read_frame(next_index)
        except IndexError:
            raise StopIteration

    def reset(self):
        """Reset internal position so next read_next() returns frame 0."""
        self.current_index = -1

# def load_from_json(path, frame=None):
#     """Load data from a JSON or JSONL file and return (landmarks, pose, cov).

#     Supported input shapes (robust):
#         - JSONL: multiple JSON objects, each with keys 'robot_position' and 'map'.
#             By default the first non-empty line is used as the initial state (can be
#             overridden with the --frame CLI option).
#     - JSON: a single JSON object with keys 'robot_position' (or 'pose') and
#       'map' or 'landmarks'.

#     Returns:
#       landmarks: list of (id, x, y)
#       pose: (x, y, theta)
#       cov: 2x2 numpy array (default if not present in file)
#     """
#     # read whole file first; support JSONL by picking the last non-empty line
#     with open(path, 'r', encoding='utf-8') as f:
#         text = f.read()

#     text = text.strip()
#     if not text:
#         raise ValueError(f"Empty file: {path}")

#     data = None
#     # detect JSONL (multiple lines / multiple top-level JSON objects)
#     if '\n' in text:
#         # collect non-empty JSON lines
#         lines = [ln for ln in text.splitlines() if ln.strip()]
#         if not lines:
#             raise ValueError(f"No JSON objects found in {path}")

#         # decide which frame to use
#         chosen = None
#         # default behaviour: treat None as 'first'
#         if frame is None or str(frame).lower() == 'first':
#             chosen = lines[0]
#         elif str(frame).lower() == 'last':
#             chosen = lines[-1]
#         else:
#             # try integer index (0-based)
#             try:
#                 idx = int(frame)
#             except Exception:
#                 raise ValueError("--frame must be 'first', 'last' or an integer index (0-based)")
#             if idx < 0 or idx >= len(lines):
#                 raise IndexError(f"Frame index {idx} out of range (0..{len(lines)-1})")
#             chosen = lines[idx]

#         try:
#             data = json.loads(chosen)
#         except Exception:
#             # fallback: try to parse entire file as JSON array/object
#             try:
#                 data = json.loads(text)
#             except Exception as e:
#                 raise ValueError(f"Failed to parse JSONL or JSON from {path}: {e}")
#     else:
#         try:
#             data = json.loads(text)
#         except Exception as e:
#             raise ValueError(f"Failed to parse JSON from {path}: {e}")

#     # extract pose
#     pose = None
#     if isinstance(data, dict):
#         if 'robot_position' in data:
#             rp = data['robot_position']
#             pose = (float(rp[0]), float(rp[1]), float(rp[2]))
#         elif 'pose' in data:
#             p = data['pose']
#             # accept [x,y,theta] or dict
#             if isinstance(p, (list, tuple)):
#                 pose = (float(p[0]), float(p[1]), float(p[2]))
#             elif isinstance(p, dict):
#                 pose = (float(p.get('x', 0.0)), float(p.get('y', 0.0)), float(p.get('theta', 0.0)))

#         # landmarks / map
#         landmarks = []
#         if 'map' in data and isinstance(data['map'], (list, tuple)):
#             mp = data['map']
#             # map as list of [x,y] points
#             for i, item in enumerate(mp):
#                 try:
#                     x, y = float(item[0]), float(item[1])
#                 except Exception:
#                     # skip malformed entries
#                     continue
#                 landmarks.append((i + 1, x, y))
#         elif 'landmarks' in data:
#             lm = data['landmarks']
#             # accept list of dicts or list of [id,x,y] or [x,y]
#             for i, item in enumerate(lm):
#                 if isinstance(item, dict):
#                     _id = item.get('id', i + 1)
#                     x = item.get('x') if 'x' in item else (item.get('0') if '0' in item else None)
#                     y = item.get('y') if 'y' in item else (item.get('1') if '1' in item else None)
#                     try:
#                         landmarks.append((int(_id), float(x), float(y)))
#                     except Exception:
#                         continue
#                 elif isinstance(item, (list, tuple)):
#                     if len(item) == 3:
#                         landmarks.append((int(item[0]), float(item[1]), float(item[2])))
#                     elif len(item) == 2:
#                         landmarks.append((i + 1, float(item[0]), float(item[1])))
#         else:
#             landmarks = []

#         # covariance (optional)
#         cov = None
#         if 'cov' in data:
#             try:
#                 arr = np.array(data['cov'], dtype=float)
#                 if arr.shape == (2, 2):
#                     cov = arr
#             except Exception:
#                 cov = None

#         # fallback defaults
#         if pose is None:
#             # default pose at origin
#             pose = (0.0, 0.0, 0.0)
#         if cov is None:
#             cov = np.array([[0.5, 0.0], [0.0, 0.3]])

#         return landmarks, pose, cov

#     else:
#         raise ValueError(f"Unsupported JSON root type: {type(data)} in {path}")

# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Simple EKF-SLAM Visualizer')
    parser.add_argument('--data', '-d', help='JSON file with initial landmarks/pose', default=None)
    parser.add_argument('--frame', '-f', help="Which JSONL frame to use: 'first', 'last' (default) or a 0-based integer index", default='last')
    args = parser.parse_args()

    if args.data:
        # interpret --frame argument: allow 'first', 'last' or integer (0-based)
        frame_arg = args.frame
        try:
            # make a new DataLoader object. Pass that to the visulizer
            # viz reads from the dataloader and draws each frame.
            landmarks, pose, cov = 
        except Exception as e:
            print("Failed to load data:", e)
            sys.exit(1)
    else:
        # default demo scene
        # rng = np.random.RandomState(0)
        # landmarks = [(i+1, float(rng.uniform(-15, 15)), float(rng.uniform(-15, 15))) for i in range(10)]
        # pose = (0.0, 0.0, math.radians(20))
        # cov = np.array([[0.5, 0.0], [0.0, 0.3]])
        pass

    viz = Visualizer(landmarks=landmarks, robot_pose=pose, robot_cov=cov)
    print("Visualizer controls: space=step, r=run/pause, q=quit")
    viz.run()

if __name__ == '__main__':
    main()