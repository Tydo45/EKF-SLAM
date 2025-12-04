import json
import numpy as np

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
        """Convert a parsed JSON dict into (landmarks, pose).

        This loader intentionally does not return covariance; the visualizer
        will use its default robot covariance when necessary.
        """
        pose = None
        landmarks = []

        if not isinstance(data, dict):
            raise ValueError("Expected JSON object for frame")

        if 'robot_position' in data:
            rp = data['robot_position']
            pose = (float(rp[0]), float(rp[1]), float(rp[2]))
        elif 'pose' in data:
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
        elif 'landmarks' in data:
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

        # we intentionally ignore any 'cov' field here; visualizer has its own default
        if pose is None:
            pose = (0.0, 0.0, 0.0)

        return landmarks, pose

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
        """Read a specific frame (0-based index) and return (landmarks, pose).

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
        """Read the next frame sequentially and return (landmarks, pose).

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
