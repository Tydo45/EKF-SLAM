import math

DIST_FOR_HALF = 5

class Metrics:
    """Class to compute and store metrics for SLAM evaluation.

    Currently computes only distance-based score for landmarks.

    Attributes:
        landmark_pos_found: list of (x, y) tuples for found landmarks
        landmark_pos_true: list of (x, y) tuples for true landmarks
        total_score: cumulative score over all landmarks
    """
    def __init__(self, landmark_pos_true, landmark_pos_found):
        self.landmark_pos_found = landmark_pos_found
        self.landmark_pos_true = landmark_pos_true
        self.total_score = 0.0
        self.count = 0
        self.matches = []
    
    def score(self):
        """Compute per-landmark metrics and return the average score over matches."""
        self._compute_landmark_metrics()
        if self.count == 0:
            self.avg_score = 0.0
        else:
            self.avg_score = self.total_score / float(self.count)
        return self.avg_score
    
    def _compute_dist_score(self, dist):
        """Compute a score based on distance.

        The score is in [0, 100], with 100 being at distance 0, and
        approaching 0 as distance increases.

        Args:
            dist: distance value (float)
        Returns:
            score: computed score (float)
        """
        score = 1/((dist*0.01/DIST_FOR_HALF)+0.01)
        if score < 0:
            score = 0.0
        return score

    def _compute_landmark_metrics(self):
        """For each found landmark, find nearest true landmark and record distance/score."""

        self.total_score = 0.0
        self.count = 0
        self.matches = []  # list of dicts: {'found_idx', 'true_idx', 'dist', 'score'}

        if not self.landmark_pos_found or not self.landmark_pos_true:
            return

        for i, (fx, fy) in enumerate(self.landmark_pos_found):
            # compute distances to all true landmarks and pick the nearest
            min_dist = float('inf')
            min_j = None
            for j, (tx, ty) in enumerate(self.landmark_pos_true):
                d = math.hypot(fx - tx, fy - ty)
                if d < min_dist:
                    min_dist = d
                    min_j = j

            score = self._compute_dist_score(min_dist)
            self.total_score += score
            self.count += 1
            self.matches.append({
                'found_idx': i,
                'true_idx': min_j,
                'dist': min_dist,
                'score': score
            })
    