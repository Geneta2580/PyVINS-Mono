import bisect
import threading

import numpy as np


class GlobalMap:
    """保存已经离开滑窗的帧位姿和路标。所有读写都在同一把锁内完成。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.frames = {}
        self.landmarks = {}
        self.keyframe_ids = set()
        self.landmark_reference_frames = {}

    def add_frame(self, frame_id, pose, is_keyframe=False):
        if pose is None:
            return
        copied = np.asarray(pose, dtype=float).reshape(4, 4).copy()
        with self._lock:
            frame_id = int(frame_id)
            self.frames[frame_id] = copied
            if is_keyframe:
                self.keyframe_ids.add(frame_id)

    def add_landmarks(self, landmarks, reference_frame_ids=None):
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
            if reference_frame_ids:
                for landmark_id in copied:
                    reference_id = reference_frame_ids.get(landmark_id)
                    if reference_id is not None:
                        self.landmark_reference_frames[landmark_id] = int(reference_id)

    def pose_graph_snapshot(self):
        """Return only archived keyframes; callers may append active keyframes."""
        with self._lock:
            return {
                frame_id: self.frames[frame_id].copy()
                for frame_id in self.keyframe_ids
                if frame_id in self.frames
            }

    def apply_pose_graph_update(self, input_poses, optimized_poses, T_bc):
        """Atomically apply per-KF corrections to all archived map entities."""
        # Kept in the interface because pose-graph poses are body poses. A
        # world-frame correction left-multiplies both body and camera poses,
        # so the fixed body-camera extrinsic cancels for world landmarks.
        np.asarray(T_bc, dtype=float).reshape(4, 4)
        input_poses = {
            int(frame_id): np.asarray(pose, dtype=float).reshape(4, 4)
            for frame_id, pose in input_poses.items()
        }
        optimized_poses = {
            int(frame_id): np.asarray(pose, dtype=float).reshape(4, 4)
            for frame_id, pose in optimized_poses.items()
        }
        correction_ids = sorted(set(input_poses) & set(optimized_poses))
        if not correction_ids:
            return
        corrections = {
            frame_id: optimized_poses[frame_id] @ np.linalg.inv(input_poses[frame_id])
            for frame_id in correction_ids
        }

        def correction_for(reference_id):
            # Frame IDs are chronological. Non-keyframes and entities archived
            # while PGO was running follow the closest preceding keyframe; an
            # entity before the first graph node follows that first node.
            reference_id = int(reference_id)
            index = bisect.bisect_right(correction_ids, reference_id) - 1
            selected_id = correction_ids[max(index, 0)]
            return corrections[selected_id]

        with self._lock:
            for frame_id, pose in list(self.frames.items()):
                if frame_id in optimized_poses:
                    self.frames[frame_id] = optimized_poses[frame_id].copy()
                else:
                    self.frames[frame_id] = correction_for(frame_id) @ pose

            for landmark_id, position in list(self.landmarks.items()):
                reference_id = self.landmark_reference_frames.get(landmark_id)
                if reference_id is None:
                    continue
                correction = correction_for(reference_id)
                self.landmarks[landmark_id] = (
                    correction[:3, :3] @ position + correction[:3, 3]
                )

    def snapshot(self):
        with self._lock:
            frames = {frame_id: pose.copy() for frame_id, pose in self.frames.items()}
            landmarks = {
                landmark_id: position.copy()
                for landmark_id, position in self.landmarks.items()
            }
        return frames, landmarks
