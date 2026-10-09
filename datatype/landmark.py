from enum import Enum
import numpy as np

class LandmarkStatus(Enum):
    CANDIDATE = 0  # 候选点，尚未被三角化
    TRIANGULATED = 1 # 已三角化，有3D位置

class Landmark:
    def __init__(self, landmark_id, first_frame_id, first_pt_2d):
        self.id = landmark_id
        
        # 初始状态为候选点，还没有3D位置
        self.status = LandmarkStatus.CANDIDATE
        self.position_3d = None
        
        # 首个观测帧。旧帧离开后，这个标签转到剩余最早观测，世界坐标 Point3 保持不变。
        self.host_frame_id = first_frame_id

        # 记录所有的观测 {frame_id: pt_2d_coords}
        self.observations = {first_frame_id: first_pt_2d}

    def add_observation(self, frame_id, pt_2d):
        self.observations[frame_id] = pt_2d

    def remove_observation(self, frame_id):
        if frame_id in self.observations:
            del self.observations[frame_id]

    def get_observation_count(self):
        return len(self.observations)

    def get_observing_frame_ids(self):
        return self.observations.keys()

    def get_observation(self, frame_id):
        return self.observations[frame_id]

    def set_triangulated(self, position_3d):
        self.position_3d = position_3d
        self.status = LandmarkStatus.TRIANGULATED

    def is_ready_for_triangulation(self, frame_window, parallax_threshold):
        # 必须是候选点，且至少有三个观测
        if self.status != LandmarkStatus.CANDIDATE or self.get_observation_count() < 3:
            return False, None, None

        # 找到第一个和最后一个观测它的、且仍在滑动窗口内的帧
        obs_ids = list(self.observations.keys())
        first_frame_id = min(obs_ids)
        last_frame_id = max(obs_ids)

        first_frame = next((frame for frame in frame_window if frame.get_id() == first_frame_id), None)
        last_frame = next((frame for frame in frame_window if frame.get_id() == last_frame_id), None)

        if first_frame is None or last_frame is None or first_frame_id == last_frame_id:
            return False, None, None

        # 检查视差
        pt1 = self.observations[first_frame_id]
        pt2 = self.observations[last_frame_id]
        parallax = np.linalg.norm(pt1 - pt2)

        if parallax > parallax_threshold:
            return True, first_frame, last_frame
        else:
            # print(f"[Trace l{self.id}]: FAILED triangulation check. Parallax: {parallax:.2f}px")
            return False, None, None
