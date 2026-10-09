import threading

import numpy as np


class GlobalMap:
    """保存已经离开滑窗的帧位姿和路标。所有读写都在同一把锁内完成。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.frames = {}
        self.landmarks = {}

    def add_frame(self, frame_id, pose):
        if pose is None:
            return
        copied = np.asarray(pose, dtype=float).reshape(4, 4).copy()
        with self._lock:
            self.frames[int(frame_id)] = copied

    def add_landmarks(self, landmarks):
        if not landmarks:
            return
        copied = {}
        for landmark_id, position in landmarks.items():
            if position is None:
                continue
            copied[int(landmark_id)] = np.asarray(position, dtype=float).reshape(3).copy()
        if not copied:
            return
        with self._lock:
            self.landmarks.update(copied)

    def snapshot(self):
        with self._lock:
            frames = {frame_id: pose.copy() for frame_id, pose in self.frames.items()}
            landmarks = {
                landmark_id: position.copy()
                for landmark_id, position in self.landmarks.items()
            }
        return frames, landmarks
