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

def load_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    landmarks = []
    for lm in data.get('landmarks', []):
        landmarks.append((lm.get('id', len(landmarks)+1), lm['x'], lm['y']))
    pose = tuple(data.get('robot_pose', (0.0, 0.0, 0.0)))
    cov = np.array(data.get('robot_covariance', [[0.2, 0.0], [0.0, 0.2]]))
    return landmarks, pose, cov

# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Simple EKF-SLAM Visualizer')
    parser.add_argument('--data', '-d', help='JSON file with initial landmarks/pose', default=None)
    args = parser.parse_args()

    if args.data:
        try:
            landmarks, pose, cov = load_from_json(args.data)
        except Exception as e:
            print("Failed to load data:", e)
            sys.exit(1)
    else:
        # default demo scene
        rng = np.random.RandomState(0)
        landmarks = [(i+1, float(rng.uniform(-15, 15)), float(rng.uniform(-15, 15))) for i in range(10)]
        pose = (0.0, 0.0, math.radians(20))
        cov = np.array([[0.5, 0.0], [0.0, 0.3]])

    viz = Visualizer(landmarks=landmarks, robot_pose=pose, robot_cov=cov)
    print("Visualizer controls: space=step, r=run/pause, q=quit")
    viz.run()

if __name__ == '__main__':
    main()