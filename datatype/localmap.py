from collections import deque
from pickle import TRUE
from datatype.landmark import Landmark, LandmarkStatus
import numpy as np
import cv2
import gtsam
import time

class LocalMap:
    def __init__(self, config):
        self.config = config

        # 读取外参
        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)

        self.max_depth = self.config.get('max_depth', 400)
        self.min_depth = self.config.get('min_depth', 0.4)
        self.triangulation_max_reprojection_error = self.config.get('triangulation_max_reprojection_error', 60.0)
        self.optimization_max_reprojection_error = self.config.get('optimization_max_reprojection_error', 60.0)
        self.optimization_max_delete_reprojection_error = self.config.get('optimization_max_delete_reprojection_error', 1000.0)
        self.min_parallax_angle_deg = self.config.get('min_parallax_angle_deg', 5.0)

        self.cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)

        # 使用字典来存储，方便通过ID快速访问
        self.frames = {}  # {frame_id: Frame}
        self.landmarks = {}  # {lm_id: Landmark_Object}

    def add_frame(self, frame):
        self.frames[frame.get_id()] = frame

        # 更新Landmark的观测信息，或创建新的Landmark，创建后默认为CANDIDATE
        for lm_id, pt_2d in zip(frame.get_visual_feature_ids(), frame.get_visual_features()):
            if lm_id in self.landmarks:
                self.landmarks[lm_id].add_observation(frame.get_id(), pt_2d)
            else:
                new_lm = Landmark(lm_id, frame.get_id(), pt_2d)
                self.landmarks[lm_id] = new_lm

    def remove_frame(self, frame_id, transfer_host=False):
        """删除指定帧及其观测。普通帧边缘化时把 host 转到剩余最早观测，不改世界坐标。"""
        if frame_id not in self.frames:
            return []

        print(f"【LocalMap】: Removing frame {frame_id}, transfer_host={transfer_host}.")
        del self.frames[frame_id]

        empty_host_ids = []
        for landmark in self.landmarks.values():
            landmark.remove_observation(frame_id)
            if not transfer_host or landmark.host_frame_id != frame_id:
                continue
            remaining_ids = sorted(landmark.observations.keys())
            if remaining_ids:
                landmark.host_frame_id = remaining_ids[0]
            else:
                empty_host_ids.append(landmark.id)

        for lm_id in empty_host_ids:
            self.landmarks.pop(lm_id, None)

        stale_lm_ids = self.prune_stale_landmarks()
        removed_ids = list(empty_host_ids)
        if stale_lm_ids:
            removed_ids.extend(stale_lm_ids)
        return removed_ids

    def prune_stale_landmarks(self):
        active_landmark_ids = set()
        for frame in self.frames.values():
            active_landmark_ids.update(frame.get_visual_feature_ids())

        stale_ids = [lm_id for lm_id in self.landmarks if lm_id not in active_landmark_ids]
        
        if stale_ids:
            print(f"【LocalMap】: Pruning {len(stale_ids)} stale landmarks.")
            print(f"【LocalMap】: Stale landmarks: {stale_ids}")
            for lm_id in stale_ids:
                del self.landmarks[lm_id]
            
            return stale_ids
        
        return None

    def get_active_frames(self):
        # 按 ID 排序返回当前窗口中的全部帧
        return sorted(self.frames.values(), key=lambda frame: frame.get_id())
    
    def get_active_landmarks(self):
        return {lm.id: lm.position_3d for lm in self.landmarks.values() if lm.status == LandmarkStatus.TRIANGULATED}

    def get_candidate_landmarks(self):
        return [lm for lm in self.landmarks.values() if lm.status == LandmarkStatus.CANDIDATE]

    def check_landmark_health(self, landmark_id, candidate_position_3d=None):
        lm = self.landmarks.get(landmark_id)
        # 必须是已三角化的点才有3D位置
        if not lm:
            return False

        # 对于还没有确认三角化的点，使用候选位置
        if candidate_position_3d is not None:
            landmark_pos = candidate_position_3d
        # 对于已经三角化的点，使用三角化后的位置
        elif lm.status == LandmarkStatus.TRIANGULATED and lm.position_3d is not None:
            landmark_pos = lm.position_3d
        else:
            return False

        observing_frame_ids = lm.get_observing_frame_ids()
        witness_frames = [self.frames[frame_id] for frame_id in observing_frame_ids if frame_id in self.frames]

        # 至少需要2个观测帧
        if len(witness_frames) < 3:
            return False
            
        positions = []
        for frame in witness_frames:
            T_w_b = frame.get_global_pose()
            T_w_c = T_w_b @ self.T_bc
            if T_w_c is not None:
                positions.append(T_w_c[:3, 3])

        if len(positions) < 3:
            return False
            
        positions = np.array(positions)

        # 计算观测基线
        baseline = np.linalg.norm(np.ptp(positions, axis=0))

        # # 基线太短，排除
        # if baseline < 0.05:
        #     print(f"【Health Check】: Landmark {lm.id} failed baseline check. Baseline: {baseline:.4f}m")
        #     return False

        # 计算路标点到观测中心的大致深度
        avg_cam_pos = np.mean(positions, axis=0) # 观测中心
        depth = np.linalg.norm(landmark_pos - avg_cam_pos)

        # 避免除以零
        if depth < 1e-6:
            return False
        
        # 检查基线与深度的比值（近似于 2 * tan(parallax_angle / 2)）
        # 一个小的角度，tan(theta)约等于theta（弧度）
        ratio = baseline / depth
        threshold = np.deg2rad(self.min_parallax_angle_deg)

        print(f"【Triangulation Health Check】: Landmark {lm.id} ratio: {ratio:.4f}, threshold: {threshold:.4f}")
        if ratio < threshold:
            print(f"【Triangulation Health Check】: Landmark {lm.id} failed parallax check. theta: {ratio:.4f}")
            return False

        # 检查重投影误差和深度
        reproj_error_total = 0.0
        for frame in witness_frames:
            T_w_b = frame.get_global_pose()
            if T_w_b is None: continue

            # 转换到相机坐标系下
            T_w_c = T_w_b @ self.T_bc
            T_c_w = np.linalg.inv(T_w_c)
            point_in_cam_homo = T_c_w @ np.append(landmark_pos, 1.0)
            
            # 深度必须为正
            depth = point_in_cam_homo[2] / point_in_cam_homo[3]
            print(f"【Triangulation Health Check】: Landmark {lm.id} depth: {depth:.4f}")
            if depth <= self.min_depth or depth > self.max_depth:
                print(f"【Triangulation Health Check】: Landmark {lm.id} failed cheirality in frame {frame.get_id()}. Depth: {depth:.4f}m")
                return False

            # 检查重投影误差
            rvec, _ = cv2.Rodrigues(T_c_w[:3,:3])
            tvec = T_c_w[:3,3]
            reprojected_pt, _ = cv2.projectPoints(landmark_pos.reshape(1,1,3), rvec, tvec, self.cam_intrinsics, None)
            reproj_error = np.linalg.norm(reprojected_pt.flatten() - lm.observations[frame.get_id()])
            reproj_error_total += reproj_error

        reproj_error_avg = reproj_error_total / len(witness_frames)
        if reproj_error_avg > self.triangulation_max_reprojection_error:
            print(f"【Triangulation Health Check】: Landmark {lm.id} failed reprojection in frame {frame.get_id()}. Error: {reproj_error_avg:.2f}px")
            return False

        return True

    
    def check_landmark_health_after_optimization(self, landmark_id):
        lm = self.landmarks.get(landmark_id)
        # 必须是已三角化的点才有3D位置
        if not lm or lm.position_3d is None:
            return False, True, True

        observing_frame_ids = [frame_id for frame_id in lm.get_observing_frame_ids() if frame_id in self.frames]
        
        # 观测帧数太少，被先验因子约束无法检查，直接返回True
        if len(observing_frame_ids) < 2:
            return True, True, True 

        # 检查全部帧
        frames_to_check = [self.frames[frame_id] for frame_id in observing_frame_ids]

        # # 优化：只检查ID最小和最大的两个观测帧
        # first_frame_id = min(observing_frame_ids)
        # last_frame_id = max(observing_frame_ids)
        
        # # 将要检查的帧限制在这两个极端
        # frames_to_check = [self.frames[first_frame_id]]
        # if first_frame_id != last_frame_id:
        #     frames_to_check.append(self.frames[last_frame_id])

        reproj_error_total = 0.0
        for frame in frames_to_check:
            T_w_b = frame.get_global_pose()
            if T_w_b is None: continue

            T_w_c = T_w_b @ self.T_bc
            T_c_w = np.linalg.inv(T_w_c)
            point_in_cam_homo = T_c_w @ np.append(lm.position_3d, 1.0)
            
            # 检查深度是否为正且在合理范围内
            depth = point_in_cam_homo[2]
            if depth <= self.min_depth or depth > self.max_depth:
                if depth < 0.0:
                    print(f"【Optimization Health Check】: Landmark {lm.id} has negative depth in frame {frame.get_id()}. Depth: {depth:.4f}m")
                    return False, False, True
                print(f"【Optimization Health Check】: Landmark {lm.id} failed depth check in frame {frame.get_id()}. Depth: {depth:.4f}m")
                return False, True, True

            # 检查重投影误差
            rvec, _ = cv2.Rodrigues(T_c_w[:3,:3])
            tvec = T_c_w[:3,3]
            reprojected_pt, _ = cv2.projectPoints(lm.position_3d.reshape(1,1,3), rvec, tvec, self.cam_intrinsics, None)
            reproj_error = np.linalg.norm(reprojected_pt.flatten() - lm.observations[frame.get_id()])
            reproj_error_total += reproj_error

        reproj_error_avg = reproj_error_total / len(frames_to_check)
        if reproj_error_avg > self.optimization_max_reprojection_error:
            if reproj_error_avg > self.optimization_max_delete_reprojection_error:
                print(f"【Optimization Health Check】: Landmark {lm.id} failed reprojection is too large in frame {frame.get_id()}. Error: {reproj_error_avg:.2f}px")
                return False, True, False
            print(f"【Optimization Health Check】: Landmark {lm.id} failed reprojection in frame {frame.get_id()}. Error: {reproj_error_avg:.2f}px")
            return False, True, True

        return True, True, True