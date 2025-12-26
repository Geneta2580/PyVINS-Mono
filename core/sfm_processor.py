import cv2
import numpy as np

class SfMProcessor:
    def __init__(self, config, cam_intrinsics):
        self.reprojection_threshold = config.get('initial_sfm_reprojection_threshold', 3.0)
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
        if len(common_ids) < 30:
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
        if num_inliers < 30:
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

        if not success:
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