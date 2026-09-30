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

        self.max_keyframes = self.config.get('window_size', 10)
        self.max_depth = self.config.get('max_depth', 400)
        self.min_depth = self.config.get('min_depth', 0.4)
        self.triangulation_max_reprojection_error = self.config.get('triangulation_max_reprojection_error', 60.0)
        self.optimization_max_reprojection_error = self.config.get('optimization_max_reprojection_error', 60.0)
        self.optimization_max_delete_reprojection_error = self.config.get('optimization_max_delete_reprojection_error', 1000.0)
        self.min_parallax_angle_deg = self.config.get('min_parallax_angle_deg', 5.0)

        self.cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)

        # 使用字典来存储，方便通过ID快速访问
        self.keyframes = {}  # {kf_id: KeyFrame_Object}
        self.landmarks = {}  # {lm_id: Landmark_Object}

    def add_keyframe(self, kf):
        self.keyframes[kf.get_id()] = kf

        suspect_lm_id = 7747 # <--- 设置我们要追踪的目标

        # 更新Landmark的观测信息，或创建新的Landmark，创建后默认为CANDIDATE
        # DEBUG
        for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
            if lm_id in self.landmarks:
                self.landmarks[lm_id].add_observation(kf.get_id(), pt_2d)
                if lm_id == suspect_lm_id:
                    print(f"🕵️‍ [Trace l{suspect_lm_id}]: OBSERVED by new KF {kf.get_id()}. Total observations: {self.landmarks[lm_id].get_observation_count()}")
            else:
                new_lm = Landmark(lm_id, kf.get_id(), pt_2d)
                self.landmarks[lm_id] = new_lm
                if lm_id == suspect_lm_id:
                    print(f"🕵️‍ [Trace l{suspect_lm_id}]: --- BORN! First seen in KF {kf.get_id()} ---")
        # DEBUG
        
        # 维护滑动窗口，剔除最老的关键帧
        if len(self.keyframes) > self.max_keyframes:
            # 找到ID最小的关键帧
            oldest_kf = min(self.keyframes.values(), key=lambda kf: kf.get_timestamp())
            oldest_kf_id = oldest_kf.get_id()
            print(f"【LocalMap】: Sliding window is full. Removing oldest KeyFrame {oldest_kf_id}.")
            del self.keyframes[oldest_kf_id]

            for landmark in self.landmarks.values():
                landmark.remove_observation(oldest_kf_id)

            # 关键帧被移除后，需要清理一下不再被观测的路标点
            stale_lm_ids = self.prune_stale_landmarks()
            return stale_lm_ids
        
        return None

    def prune_stale_landmarks(self):
        active_landmark_ids = set()
        for kf in self.keyframes.values():
            active_landmark_ids.update(kf.get_visual_feature_ids())

        stale_ids = [lm_id for lm_id in self.landmarks if lm_id not in active_landmark_ids]
        
        if stale_ids:
            print(f"【LocalMap】: Pruning {len(stale_ids)} stale landmarks.")
            print(f"【LocalMap】: Stale landmarks: {stale_ids}")
            for lm_id in stale_ids:
                del self.landmarks[lm_id]
            
            return stale_ids
        
        return None

    def get_active_keyframes(self):
        # 按ID排序后返回，确保顺序
        return sorted(self.keyframes.values(), key=lambda kf: kf.get_id())
    
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

        observing_kf_ids = lm.get_observing_kf_ids()
        witness_kfs = [self.keyframes[kf_id] for kf_id in observing_kf_ids if kf_id in self.keyframes]

        # 至少需要2个观测帧
        if len(witness_kfs) < 3:
            return False
            
        positions = []
        for kf in witness_kfs:
            T_w_b = kf.get_global_pose()
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
        for kf in witness_kfs:
            T_w_b = kf.get_global_pose()
            if T_w_b is None: continue

            # 转换到相机坐标系下
            T_w_c = T_w_b @ self.T_bc
            T_c_w = np.linalg.inv(T_w_c)
            point_in_cam_homo = T_c_w @ np.append(landmark_pos, 1.0)
            
            # 深度必须为正
            depth = point_in_cam_homo[2] / point_in_cam_homo[3]
            print(f"【Triangulation Health Check】: Landmark {lm.id} depth: {depth:.4f}")
            if depth <= self.min_depth or depth > self.max_depth:
                print(f"【Triangulation Health Check】: Landmark {lm.id} failed cheirality in KF {kf.get_id()}. Depth: {depth:.4f}m")
                return False

            # 检查重投影误差
            rvec, _ = cv2.Rodrigues(T_c_w[:3,:3])
            tvec = T_c_w[:3,3]
            reprojected_pt, _ = cv2.projectPoints(landmark_pos.reshape(1,1,3), rvec, tvec, self.cam_intrinsics, None)
            reproj_error = np.linalg.norm(reprojected_pt.flatten() - lm.observations[kf.get_id()])
            reproj_error_total += reproj_error

        reproj_error_avg = reproj_error_total / len(witness_kfs)
        if reproj_error_avg > self.triangulation_max_reprojection_error:
            print(f"【Triangulation Health Check】: Landmark {lm.id} failed reprojection in KF {kf.get_id()}. Error: {reproj_error_avg:.2f}px")
            return False

        return True

    
    def check_landmark_health_after_optimization(self, landmark_id):
        lm = self.landmarks.get(landmark_id)
        # 必须是已三角化的点才有3D位置
        if not lm or lm.position_3d is None:
            return False, True, True

        observing_kf_ids = [kf_id for kf_id in lm.get_observing_kf_ids() if kf_id in self.keyframes]
        
        # 观测帧数太少，被先验因子约束无法检查，直接返回True
        if len(observing_kf_ids) < 2:
            return True, True, True 

        # 检查全部KF
        kfs_to_check = [self.keyframes[kf_id] for kf_id in observing_kf_ids]

        # # 优化：只检查ID最小和最大的两个观测帧
        # first_kf_id = min(observing_kf_ids)
        # last_kf_id = max(observing_kf_ids)
        
        # # 将要检查的关键帧限制在这两个极端
        # kfs_to_check = [self.keyframes[first_kf_id]]
        # if first_kf_id != last_kf_id:
        #     kfs_to_check.append(self.keyframes[last_kf_id])

        reproj_error_total = 0.0
        for kf in kfs_to_check:
            T_w_b = kf.get_global_pose()
            if T_w_b is None: continue

            T_w_c = T_w_b @ self.T_bc
            T_c_w = np.linalg.inv(T_w_c)
            point_in_cam_homo = T_c_w @ np.append(lm.position_3d, 1.0)
            
            # 检查深度是否为正且在合理范围内
            depth = point_in_cam_homo[2]
            if depth <= self.min_depth or depth > self.max_depth:
                if depth < 0.0:
                    print(f"【Optimization Health Check】: Landmark {lm.id} has negative depth in KF {kf.get_id()}. Depth: {depth:.4f}m")
                    return False, False, True
                print(f"【Optimization Health Check】: Landmark {lm.id} failed depth check in KF {kf.get_id()}. Depth: {depth:.4f}m")
                return False, True, True

            # 检查重投影误差
            rvec, _ = cv2.Rodrigues(T_c_w[:3,:3])
            tvec = T_c_w[:3,3]
            reprojected_pt, _ = cv2.projectPoints(lm.position_3d.reshape(1,1,3), rvec, tvec, self.cam_intrinsics, None)
            reproj_error = np.linalg.norm(reprojected_pt.flatten() - lm.observations[kf.get_id()])
            reproj_error_total += reproj_error

        reproj_error_avg = reproj_error_total / len(kfs_to_check)
        if reproj_error_avg > self.optimization_max_reprojection_error:
            if reproj_error_avg > self.optimization_max_delete_reprojection_error:
                print(f"【Optimization Health Check】: Landmark {lm.id} failed reprojection is too large in KF {kf.get_id()}. Error: {reproj_error_avg:.2f}px")
                return False, True, False
            print(f"【Optimization Health Check】: Landmark {lm.id} failed reprojection in KF {kf.get_id()}. Error: {reproj_error_avg:.2f}px")
            return False, True, True

        return True, True, True