import cv2
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

class SfMProcessor:
    def __init__(self, config, cam_intrinsics):
        self.reprojection_threshold = config.get('initial_sfm_reprojection_threshold', 3.0)
        self.initial_min_matches = int(config.get('initial_min_matches', 20))
        self.cam_intrinsics = cam_intrinsics
        self.config = config
        self.keyframes = {}
        self.last_kf = None
        self.last_kf_pts = None
        self.last_kf_gray = None

    def find_matches_features(self, kf1, kf2):
        # 将id映射为索引
        ids1_map = {fid: i for i, fid in enumerate(kf1.get_visual_feature_ids())}
        ids2_map = {fid: i for i, fid in enumerate(kf2.get_visual_feature_ids())}

        # 找到两个KF中共同的特征点
        common_ids = set(ids1_map.keys()).intersection(set(ids2_map.keys()))
        pts1 = np.array([kf1.get_visual_features()[ids1_map[fid]] for fid in common_ids])
        pts2 = np.array([kf2.get_visual_features()[ids2_map[fid]] for fid in common_ids])
        return pts1, pts2, list(common_ids)

    def epipolar_compute(self, kf1, kf2):
        pts1, pts2, common_ids = self.find_matches_features(kf1, kf2)
        if len(common_ids) < self.initial_min_matches:
            print("【VO】: Not enough matches to initialize")
            return False, None, None, None, None, None

        # 计算本质矩阵
        E, inlier_mask = cv2.findEssentialMat(
            pts1, pts2, self.cam_intrinsics, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print("【VO】: Failed to compute essential matrix")
            return False, None, None, None, None, None

        # 计算基础矩阵
        num_inliers, R, t, final_inlier_mask = cv2.recoverPose(E, pts1, pts2, self.cam_intrinsics, inlier_mask)
        if num_inliers < self.initial_min_matches:
            print("【VO】: Failed to recover pose")
            return False, None, None, None, None, None

        final_mask_bool = final_inlier_mask.ravel().astype(bool)

        # 过滤并返回内点信息，这里过滤的不能过于严格
        inlier_ids = np.array(common_ids)[final_mask_bool]
        pts1_inliers = pts1[final_mask_bool]
        pts2_inliers = pts2[final_mask_bool]

        return True, inlier_ids, pts1_inliers, pts2_inliers, R, t
        
    def triangulate_points(self, pts1, pts2, R, t):
        pose1 = np.eye(4)
        pose2 = np.eye(4)
        pose2[:3, :3] = R
        pose2[:3, 3] = t.ravel()

        proj_mat1 = self.cam_intrinsics @ pose1[:3, :]
        proj_mat2 = self.cam_intrinsics @ pose2[:3, :]

        points4d_homo = cv2.triangulatePoints(proj_mat1, proj_mat2, pts1.T, pts2.T) # 4xN

        # 第一次筛选：找到所有 w > 1e-4 的有限点
        finite_mask = np.abs(points4d_homo[3, :]) > 1e-4 
        # print(f"【VO】: finite_mask: {finite_mask}")
        # mask长度为N，与输入等长
        final_mask_for_caller = finite_mask
        points4d_finite = points4d_homo[:, finite_mask] # 4xM
        if points4d_finite.shape[1] == 0:
            return np.array([]), np.zeros_like(finite_mask, dtype=bool)

        # 转换为非齐次坐标
        points3d_finite = points4d_finite[:3] / points4d_finite[3] # 3xM

        # 第一张图深度检查
        positive_depth_mask1 = points3d_finite[2, :] > 0

        # 第二张图深度检查
        points_in_cam2 = (R @ points3d_finite) + t
        positive_depth_mask2 = points_in_cam2[2, :] > 0
        
        cheirality_mask = positive_depth_mask1 & positive_depth_mask2 # 长度为M

        # 生成最终的3D点
        final_points3d = points3d_finite[:, cheirality_mask].T # 形状(K, 3)

        # print(f"【VO】: final_points3d: {final_points3d}")
        # 更新最终的输出掩码，使其长度为N
        final_mask_for_caller[finite_mask] = cheirality_mask
        
        return final_points3d, final_mask_for_caller

    def track_with_pnp(self, landmarks, new_keyframe):
        object_points, image_points = [], []
        
        # 得到新KF的所有特征点映射
        kf_features_map = {fid: feat for fid, feat in zip(new_keyframe.get_visual_feature_ids(), new_keyframe.get_visual_features())}

        if not landmarks:
            return False, None

        # 找到landmark在new_keyframe的投影
        for landmark_id, landmark_3d in landmarks.items():
            if landmark_id in kf_features_map:
                object_points.append(landmark_3d)
                image_points.append(kf_features_map[landmark_id])

        if len(image_points) < 10:
            return False, None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(object_points), np.array(image_points), 
            self.cam_intrinsics, None
        )

        if not success or inliers is None or len(inliers) < 10:
            return False, None

        R, _ = cv2.Rodrigues(rvec)
        T_cam_world = np.eye(4)
        T_cam_world[:3, :3] = R
        T_cam_world[:3, 3] = tvec.ravel()

        T_world_cam = np.linalg.inv(T_cam_world)
        return True, T_world_cam

    
    def filter_points_by_reprojection(self, points_3d, p1_matched, p2_matched, R, t):
        if len(points_3d) == 0:
            return np.array([]), np.array([], dtype=bool)

        # 1. 投影回第一帧 (参考帧，位姿为单位矩阵)
        rvec1_ident, tvec1_zero = np.zeros(3), np.zeros(3)
        reprojected_pts1, _ = cv2.projectPoints(points_3d, rvec1_ident, tvec1_zero, self.cam_intrinsics, None)
        
        # 2. 投影回第二帧 (当前帧)
        rvec2, _ = cv2.Rodrigues(R)
        reprojected_pts2, _ = cv2.projectPoints(points_3d, rvec2, t.ravel(), self.cam_intrinsics, None)
        
        # 3. 计算误差
        error1 = np.linalg.norm(p1_matched - reprojected_pts1.reshape(-1, 2), axis=1)
        error2 = np.linalg.norm(p2_matched - reprojected_pts2.reshape(-1, 2), axis=1)
        
        # 4. 创建掩码并过滤
        reprojection_mask = (error1 < self.reprojection_threshold) & (error2 < self.reprojection_threshold)
        filtered_points_3d = points_3d[reprojection_mask]
        
        return filtered_points_3d, reprojection_mask

    def build_feature_tracks(self, frames):
        """把初始化窗口里的去畸变像素观测收成 feature_id -> [(frame_idx, uv), ...]。"""
        raw = {}
        for frame_idx, frame in enumerate(frames):
            feature_ids = frame.get_visual_feature_ids()
            features = frame.get_visual_features()
            if feature_ids is None or features is None:
                continue
            for feature_id, uv in zip(feature_ids, features):
                raw.setdefault(int(feature_id), {})[frame_idx] = np.asarray(uv, dtype=float).reshape(2)
        tracks = {}
        for feature_id, observations in raw.items():
            if len(observations) < 2:
                continue
            tracks[feature_id] = sorted(observations.items())
        return tracks

    def initialize_window(self, frames, parallax_threshold):
        """参考帧对最新帧做相对位姿，再交替 PnP / 三角化，最后做固定尺度的视觉 BA。"""
        if len(frames) < 2:
            return None
        parallax_threshold = float(parallax_threshold)
        # 构建特征匹配对
        tracks = self.build_feature_tracks(frames) 
        # 选择参考帧（视差最大，匹配点数最多），作为初始化窗口的参考帧，同时返回参考帧的位姿和最新帧的E阵分解的旋转和平移
        selected = self._select_reference_pair(frames, parallax_threshold)

        if selected is None:
            return None
        ref_idx, newest_idx, rotation, translation = selected

        # 选定初始化帧对的位姿基线
        poses = [None] * len(frames)
        poses[ref_idx] = np.eye(4)
        relative = np.eye(4)
        relative[:3, :3] = rotation
        relative[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
        poses[newest_idx] = np.linalg.inv(relative)
        initial_baseline = np.linalg.norm(poses[newest_idx][:3, 3])

        # 三角化初始化帧对的特征点，得到初始化的3D点
        landmarks = {}
        landmarks = self._triangulate_available(tracks, poses, landmarks, max_error=5.0)
        print(f"【Visual Init】: Seed triangulation kept {len(landmarks)} landmarks.")

        # 交替PnP / 三角化，直到所有帧的位姿都被优化
        for round_idx in range(len(frames) * 2):
            if all(pose is not None for pose in poses):
                break
            progressed = False
            for frame_idx, pose in enumerate(poses):
                if pose is not None:
                    continue
                success, solved = self.track_with_pnp(landmarks, frames[frame_idx])
                # 检查PnP是否成功，并且重投影误差是否小于阈值
                if not success or not self._accept_pnp(solved, landmarks, frames[frame_idx]):
                    continue
                poses[frame_idx] = solved
                progressed = True
            if not progressed:
                break
            # 对已有多帧匹配轨迹、但尚未生成三维点的特征进行补充三角化
            landmarks = self._triangulate_available(tracks, poses, landmarks, max_error=5.0)
            posed = sum(pose is not None for pose in poses)
            print(f"【Visual Init】: round {round_idx} posed {posed}/{len(poses)}, landmarks {len(landmarks)}")

        missing = [frames[idx].get_id() for idx, pose in enumerate(poses) if pose is None]
        # 检查是否有帧的位姿没有被优化
        if missing:
            print(f"【Visual Init】: PnP left frames without pose: {missing}")
            return None
        landmarks = self._triangulate_available(tracks, poses, landmarks, max_error=5.0)
        # 优化初始化帧对的位姿和世界点
        return self.bundle_adjust_initial_window(
            frames, tracks, poses, landmarks, ref_idx, newest_idx, initial_baseline)

    def _select_reference_pair(self, frames, parallax_threshold):
        # 选择最新帧作为参考帧
        newest_idx = len(frames) - 1
        best = None
        for ref_idx in range(newest_idx):
            # 计算本质矩阵
            success, inlier_ids, pts1, pts2, rotation, translation = self.epipolar_compute(
                frames[ref_idx], frames[newest_idx])
            if not success:
                continue
            # 计算视差
            parallax = float(np.median(np.linalg.norm(pts1 - pts2, axis=1)))
            print(
                f"  - Pair (KF {frames[ref_idx].get_id()}, newest {frames[newest_idx].get_id()}) "
                f"parallax {parallax:.2f} px, matches {len(inlier_ids)}"
            )
            # VINS-Mono initialStructure(): >20 correspondences and about
            # 30 px average parallax on EuRoC before solving relative pose.
            if parallax < parallax_threshold or len(inlier_ids) < self.initial_min_matches:
                continue
            score = (parallax, len(inlier_ids))
            # 选择最佳帧
            if best is None or score > best[0]:
                best = (score, ref_idx, rotation, translation, parallax, len(inlier_ids))
        if best is None:
            print("【Visual Init】: No reference frame has enough parallax against the newest frame.")
            return None
        _, ref_idx, rotation, translation, parallax, match_count = best
        print(
            f"【Visual Init】: Selected ref KF {frames[ref_idx].get_id()} <-> "
            f"newest KF {frames[newest_idx].get_id()}, parallax {parallax:.2f} px, matches {match_count}"
        )
        return ref_idx, newest_idx, rotation, translation

    def _triangulate_available(self, tracks, poses, previous, max_error):
        updated = {}
        for feature_id, observations in tracks.items():
            posed = [(frame_idx, uv) for frame_idx, uv in observations if poses[frame_idx] is not None]
            if len(posed) < 2:
                continue
            point = self.triangulate_track(posed, poses, max_error)
            if point is not None:
                updated[feature_id] = point
            elif feature_id in previous:
                updated[feature_id] = previous[feature_id]
        return updated

    def triangulate_track(self, observations, poses, max_error,
                          min_depth=None, max_depth=None, min_parallax_angle_deg=None):
        """利用多观测帧的位姿和像素坐标计算世界点。深度区间和视差角只在调用方传入时生效。"""
        rows = []
        valid = []
        for frame_idx, uv in observations:
            pose_w_c = poses[frame_idx]
            if pose_w_c is None:
                continue
            projection = self.cam_intrinsics @ np.linalg.inv(pose_w_c)[:3, :]
            u, v = np.asarray(uv, dtype=float).reshape(2)
            rows.append(u * projection[2] - projection[0])
            rows.append(v * projection[2] - projection[1])
            valid.append((frame_idx, np.asarray(uv, dtype=float).reshape(2)))
        if len(valid) < 2:
            return None

        # 使用SVD分解计算最小二乘解并计算重投影误差
        _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float), full_matrices=False)
        homogeneous = vt[-1]
        if abs(homogeneous[3]) < 1e-9:
            return None
        point = homogeneous[:3] / homogeneous[3]
        camera_positions = []
        for frame_idx, uv in valid:
            pose = poses[frame_idx]
            camera_point = np.linalg.inv(pose)[:3, :] @ np.append(point, 1.0)
            depth = float(camera_point[2])
            if depth <= 1e-6:
                return None
            if min_depth is not None and depth <= float(min_depth):
                return None
            if max_depth is not None and depth > float(max_depth):
                return None
            if self._reprojection_error(point, pose, uv) > max_error:
                return None
            camera_positions.append(np.asarray(pose[:3, 3], dtype=float).reshape(3))
        if min_parallax_angle_deg is not None and not self._parallax_sufficient(
                point, camera_positions, min_parallax_angle_deg):
            return None
        return point

    def _parallax_sufficient(self, point, camera_positions, min_parallax_angle_deg):
        if len(camera_positions) < 3:
            return False
        positions = np.asarray(camera_positions, dtype=float)
        baseline = np.linalg.norm(np.ptp(positions, axis=0))
        center_depth = np.linalg.norm(point - np.mean(positions, axis=0))
        if center_depth < 1e-6:
            return False
        return baseline / center_depth >= np.deg2rad(float(min_parallax_angle_deg))

    def _triangulate_one(self, pose_w_a, uv_a, pose_w_b, uv_b):
        pose_b_a = np.linalg.inv(pose_w_b) @ pose_w_a
        points, _ = self.triangulate_points(
            np.asarray(uv_a, dtype=float).reshape(1, 2),
            np.asarray(uv_b, dtype=float).reshape(1, 2),
            pose_b_a[:3, :3],
            pose_b_a[:3, 3].reshape(3, 1),
        )
        if len(points) == 0:
            return None
        return pose_w_a[:3, :3] @ points[0] + pose_w_a[:3, 3]

    def _reprojection_error(self, point_w, pose_w_c, uv):
        camera_point = np.linalg.inv(pose_w_c)[:3, :] @ np.append(point_w, 1.0)
        if camera_point[2] <= 1e-6:
            return np.inf
        projected = self.cam_intrinsics @ camera_point
        projected = projected[:2] / projected[2]
        return float(np.linalg.norm(projected - np.asarray(uv, dtype=float).reshape(2)))

    def _accept_pnp(self, pose, landmarks, frame):
        feature_map = {
            int(feature_id): np.asarray(uv, dtype=float).reshape(2)
            for feature_id, uv in zip(frame.get_visual_feature_ids(), frame.get_visual_features())
        }
        errors = []
        for feature_id, point in landmarks.items():
            if feature_id not in feature_map:
                continue
            errors.append(self._reprojection_error(
                np.asarray(point, dtype=float).reshape(3), pose, feature_map[feature_id]))
        finite = [error for error in errors if np.isfinite(error)]
        return len(finite) >= 10 and float(np.median(finite)) <= 8.0

    def bundle_adjust_initial_window(
            self, frames, tracks, initial_poses, initial_landmarks, ref_idx, newest_idx, initial_baseline):
        """优化 T_wc 和世界点。参考帧位姿与最新帧平移固定，用来消掉单目 7 自由度。"""
        long_tracks = [
            feature_id for feature_id, observations in tracks.items()
            if feature_id in initial_landmarks and len(observations) >= 3
        ]
        if len(long_tracks) < 30:
            long_tracks = [
                feature_id for feature_id in initial_landmarks
                if feature_id in tracks and len(tracks[feature_id]) >= 2
            ]
        if len(long_tracks) < 30:
            print(f"【Visual BA】: Only {len(long_tracks)} tracks available.")
            return None

        calibration = gtsam.Cal3_S2(
            self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1], self.cam_intrinsics[0, 1],
            self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2])
        noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(2.0),
            gtsam.noiseModel.Isotropic.Sigma(2, 1.5))
        body_P_sensor = gtsam.Pose3()
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        for frame_idx, pose in enumerate(initial_poses):
            values.insert(X(frame_idx), gtsam.Pose3(pose))
        for feature_id in long_tracks:
            values.insert(L(feature_id), gtsam.Point3(np.asarray(initial_landmarks[feature_id], dtype=float)))
            for frame_idx, uv in tracks[feature_id]:
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    gtsam.Point2(float(uv[0]), float(uv[1])),
                    noise, X(frame_idx), L(feature_id), calibration, body_P_sensor))

        ref_pose = values.atPose3(X(ref_idx))
        graph.add(gtsam.PriorFactorPose3(
            X(ref_idx), ref_pose, gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))))
        newest_translation = np.asarray(values.atPose3(X(newest_idx)).translation(), dtype=float).reshape(3)
        graph.add(gtsam.GPSFactor(
            X(newest_idx), gtsam.Point3(newest_translation),
            gtsam.noiseModel.Isotropic.Sigma(3, 1e-4)))

        initial_error = graph.error(values)
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(30)
        try:
            result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
        except Exception as exc:
            print(f"【Visual BA】: optimization failed: {exc}")
            return None
        final_error = graph.error(result)
        print(f"【Visual BA】: cost {initial_error:.3f} -> {final_error:.3f}")
        if final_error > initial_error * 1.01 + 1e-6:
            print("【Visual BA】: cost increased. Reject this visual structure.")
            return None

        poses = [result.atPose3(X(frame_idx)).matrix() for frame_idx in range(len(frames))]
        kept_landmarks = {}
        kept_tracks = {}
        errors = []
        positive_depth = 0
        considered = 0
        for feature_id, observations in tracks.items():
            if feature_id in long_tracks and result.exists(L(feature_id)):
                point = np.asarray(result.atPoint3(L(feature_id)), dtype=float).reshape(3)
            elif feature_id in initial_landmarks:
                point = self.triangulate_track(observations, poses, max_error=5.0)
                if point is None:
                    continue
            else:
                continue
            considered += 1
            depths_positive = True
            valid_observations = []
            for frame_idx, uv in observations:
                camera_point = np.linalg.inv(poses[frame_idx])[:3, :] @ np.append(point, 1.0)
                if camera_point[2] <= 1e-6:
                    depths_positive = False
                    continue
                error = self._reprojection_error(point, poses[frame_idx], uv)
                if error <= 3.0:
                    valid_observations.append((frame_idx, uv))
            if depths_positive and len(valid_observations) == len(observations):
                positive_depth += 1
            if len(valid_observations) < 2:
                continue
            if len(valid_observations) < len(observations):
                point = self.triangulate_track(valid_observations, poses, max_error=3.0)
                if point is None:
                    continue
            observation_errors = [
                self._reprojection_error(point, poses[frame_idx], uv)
                for frame_idx, uv in valid_observations
            ]
            if any((not np.isfinite(error)) or error > 3.0 for error in observation_errors):
                continue
            kept_landmarks[feature_id] = point
            kept_tracks[feature_id] = valid_observations
            errors.extend(observation_errors)

        min_landmarks = int(self.config.get('init_min_landmarks', 50))
        positive_ratio = positive_depth / considered if considered else 0.0
        mean_error = float(np.mean(errors)) if errors else np.inf
        median_error = float(np.median(errors)) if errors else np.inf
        final_baseline = np.linalg.norm(poses[newest_idx][:3, 3] - poses[ref_idx][:3, 3])
        print(
            f"【Visual BA】: landmarks {len(kept_landmarks)} positive-depth {positive_ratio:.3f} "
            f"reproj mean/median {mean_error:.3f}/{median_error:.3f} px "
            f"baseline {initial_baseline:.4f} -> {final_baseline:.4f}"
        )
        if len(kept_landmarks) <= min_landmarks:
            print(f"【Visual BA】: landmark count {len(kept_landmarks)} <= {min_landmarks}.")
            return None
        if positive_ratio <= 0.90 or mean_error >= 1.5 or median_error >= 1.0:
            print("【Visual BA】: structure quality below the initialization gate.")
            return None
        if initial_baseline > 1e-6 and final_baseline < 0.5 * initial_baseline:
            print("【Visual BA】: baseline collapsed.")
            return None
        for frame_idx, frame in enumerate(frames):
            frame.set_global_pose(poses[frame_idx])
        return {
            'poses': poses,
            'landmarks': kept_landmarks,
            'tracks': kept_tracks,
            'ref_idx': ref_idx,
            'newest_idx': newest_idx,
        }
