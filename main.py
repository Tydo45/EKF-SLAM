import sys
import math
import argparse
import json
import numpy as np
from matplotlib.patches import Ellipse
from dataloader import DataLoader

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

ROBOT_COV = np.array([[0.02, 0.0], [0.0, 0.02]])
# for the arrow
HEAD_WIDTH=0.1
HEAD_LENGTH=0.1
ARROW_LENGTH=0.3

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
    def __init__(self, DataLoader):
        # reader for JSONL data
        self.data_loader = DataLoader
        # read initial frame
        try:
            self.data_loader.read_next() # don't want to grab the first line which has the true landmark positions
            self.landmarks, self.pose = self.data_loader.read_next()
            self.true_landmarks, _ = self.data_loader.read_frame(0)
            # DataLoader no longer returns covariance; use visualizer default
            self.cov = ROBOT_COV
        except StopIteration:
            self.landmarks = []
            self.pose = (0.0, 0.0, 0.0)
            self.cov = ROBOT_COV

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
        self.true_landmark_sc = self.ax.scatter([], [], c='red', s=50, zorder=2)
        self.landmark_sc = self.ax.scatter([], [], c='tab:green', s=50, zorder=2)
        self.landmark_texts = []
        self.true_landmark_texts = []
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
    
    def _draw_true_landmarks(self):
        if self.true_landmark_texts:
            for t in self.true_landmark_texts:
                t.remove()
            self.true_landmark_texts = []
        if self.true_landmarks:
            xs = [x for (_id, x, y) in self.true_landmarks]
            ys = [y for (_id, x, y) in self.true_landmarks]
            self.true_landmark_sc.set_offsets(np.column_stack((xs, ys)))
            # for (_id, x, y) in self.true_landmarks:
            #     txt = self.ax.text(x, y, f'{_id}', color='k', fontsize=9,
            #                        verticalalignment='bottom', horizontalalignment='right')
            #     self.true_landmark_texts.append(txt)

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
        dx = math.cos(theta) * ARROW_LENGTH
        dy = math.sin(theta) * ARROW_LENGTH
        self.robot_arrow = self.ax.arrow(x, y, dx, dy, head_width=HEAD_WIDTH, head_length=HEAD_LENGTH,
                                         fc='tab:blue', ec='tab:blue', zorder=3)
        # covariance ellipse (use xy submatrix)
        width, height, angle = cov_ellipse_params(self.cov)
        self.cov_ellipse = Ellipse((x, y), width, height, angle=angle,
                                   edgecolor='tab:blue', facecolor='tab:blue', alpha=0.15, zorder=1)
        self.ax.add_patch(self.cov_ellipse)

    def update(self):
        """Redraw everything (call after updating self.pose/self.landmarks/self.cov)."""
        self._update_plot_limits()
        self._draw_true_landmarks()
        self._draw_landmarks()
        self._draw_robot()
        self.fig.canvas.draw_idle()

    # --- read next frame from JSONL ------------
    def step_simulation(self):
        """Read next frame from JSONL data loader and update pose and landmarks.

        Note: DataLoader does not return covariance; the visualizer keeps its
        own robot covariance (`ROBOT_COV`)."""
        try:
            if (self.data_loader.current_index == 0):
                self.data_loader.current_index = 1
            self.landmarks, self.pose = self.data_loader.read_next()
            self.cov = ROBOT_COV
        except StopIteration:
            print("End of data reached.")
            self.running = False

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

# DataLoader moved to dataloader.py; imported at top of file.

# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Simple EKF-SLAM Visualizer')
    parser.add_argument('--data', '-d', help='JSON file with initial landmarks/pose', default=None)
    parser.add_argument('--frame', '-f', help="Which JSONL frame to use: 'first', 'last' (default) or a 0-based integer index", default='last')
    args = parser.parse_args()

    # interpret --frame argument: allow 'first', 'last' or integer (0-based)
    frame_arg = args.frame
    try:
        # make a new DataLoader object. Pass that to the visulizer
        # viz reads from the dataloader and draws each frame.
        dl = DataLoader(args.data)
        viz = Visualizer(dl)
        # landmarks, _ = dl.read_frame(-1)
        # true_landmarks, _ = dl.read_frame(0)
        print("Visualizer controls: space=step, r=run/pause, q=quit")
        viz.run()
    except Exception as e:
        print("Failed to load data:", e)
        sys.exit(1)



if __name__ == '__main__':
    main()