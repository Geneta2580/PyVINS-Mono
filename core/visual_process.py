import cv2
import numpy as np
import time

class VisualProcessor:
    def __init__(self, config):
        # 前端切换关键帧参数
        self.config = config
        self.max_features_to_detect = self.config.get('max_features_to_detect', 500) # 最大特征点数
        self.min_parallax = self.config.get('min_parallax', 10) # 最小视差
        self.min_stationary_parallax = self.config.get('min_stationary_parallax', 3.0) # 最小静止视差
        self.min_track_ratio = self.config.get('min_track_ratio', 0.8) # 最小跟踪比例
        self.visualize_flag = self.config.get('visualize', True) # 是否可视化追踪结果

        # 读取相机内参
        cam_matrix_raw = self.config.get('cam_intrinsics', np.eye(3).flatten().tolist())
        self.cam_matrix = np.asarray(cam_matrix_raw).reshape(3, 3)

        # 读取相机畸变参数
        self.use_undistort = self.config.get('use_undistort', True)
        self.distortion_model = self.config.get('distortion_model', 'radtan').lower()
        dist_coeffs_raw = self.config.get('distortion_coefficients', [])
        self.dist_coeffs = np.asarray(dist_coeffs_raw).astype(np.float32)

        if self.distortion_model == 'fisheye':
            # cv2.fisheye 要求畸变系数通常为 4个
            self.dist_coeffs = self.dist_coeffs.reshape(4)
        else:
            # cv2.undistortPoints 接受 (N,) 或 (1, N)
            self.dist_coeffs = self.dist_coeffs.reshape(-1)

        # 特征点间的最小像素距离
        self.min_dist = self.config.get('min_dist', 30)

        # 特征点id
        self.next_feature_id = 0
        self.prev_pt_ids = None

        self.prev_pts = None
        self.prev_gray = None

        # 特征点追踪时间
        self.feature_ages = {}
        self.long_track_age_threshold = self.config.get('long_track_age_threshold', 5)  # 定义"长追踪点"的age阈值
        self.min_long_track_ratio = self.config.get('min_long_track_ratio', 0.3)  # 长追踪点的最小比例
        self.max_age_for_color = self.config.get('max_age_for_color', 10)  # 修复：添加这行

        # 设置mask
        self.mask = None

        # RANSAC配置
        self.use_ransac = self.config.get('use_ransac', True)
        self.ransac_threshold = self.config.get('ransac_threshold', 1.0)  # 像素阈值
        self.ransac_prob = self.config.get('ransac_prob', 0.999)  # RANSAC置信度

    # 提取特征点
    def detect_features(self, gray_image, max_corners, mask=None):
        return cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=self.min_dist,
            blockSize=7,
            mask=mask
        )

    # 特征点去畸变
    def undistort_points(self, points):
        if self.cam_matrix is None or self.dist_coeffs is None or len(points) == 0:
            return points 
        
        # cv2 需要 (N, 1, 2) 的形状
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        
        # P=self.cam_matrix 表示去畸变后仍投影回原相机内参的像素坐标系
        # 如果 P=None 或 identity，则返回归一化平面坐标
        
        undistorted_points = cv2.undistortPoints(points_reshaped, self.cam_matrix, self.dist_coeffs, P=self.cam_matrix)
        
        return undistorted_points.reshape(-1, 2)

    # 按照特征点追踪时间重排并添加mask
    def filter_features_by_age(self, features, features_ids, image_shape):
        if len(features) == 0:
            empty_mask = np.ones(image_shape, dtype=np.uint8)
            return np.empty((0, 1, 2)), np.array([]), empty_mask

        features_reshaped = features.reshape(-1, 2)

        # 按照特征点的追踪时间（age）降序排序
        # 创建 (特征点, ID, age) 的列表
        pts_with_ages = []
        for pt, fid in zip(features_reshaped, features_ids):
            age = self.feature_ages.get(fid, 0)
            pts_with_ages.append((pt, fid, age))

        # 按照age降序排序
        pts_with_ages.sort(key=lambda x: x[2], reverse=True)

        # 创建mask
        age_mask = np.ones(image_shape, dtype=np.uint8)
        filtered_pts = []
        filtered_ids = []

        # 按优先级（追踪时间）保留特征点，剔除在mask范围内的低优先级点
        for pt, fid, age in pts_with_ages:
            pt_int = tuple(pt.astype(int))
            # 检查该点是否在已占用的mask区域内，需要检查坐标是否在图像范围内
            if (0 <= pt_int[1] < image_shape[0] and 0 <= pt_int[0] < image_shape[1] and 
                age_mask[pt_int[1], pt_int[0]] > 0):            
                filtered_pts.append(pt)
                filtered_ids.append(fid)
                # 在mask中标记该点已占用
                cv2.circle(age_mask, pt_int, self.min_dist, 0, -1)

        # 转换为numpy数组
        if len(filtered_pts) > 0:
            filtered_features = np.array(filtered_pts).reshape(-1, 1, 2)
            filtered_ids = np.array(filtered_ids)
        else:
            filtered_features = np.empty((0, 1, 2))
            filtered_ids = np.array([])
        
        return filtered_features, filtered_ids, age_mask


    # 光流追踪特征点
    def track_features(self, image, timestamp, predicted_pts=None, imu_data_for_prediction=None, imu_processor=None):
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 一些局部变量的初始化
        is_kf = False
        is_stationary = False
        new_pts = None

        # 记录上一帧总点数（用于计算tracking_rate）
        prev_total_count = 0 if self.prev_pt_ids is None else len(self.prev_pt_ids)

        # 第一帧处理逻辑，必定为关键帧
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_gray = curr_gray
            self.prev_pts = self.detect_features(self.prev_gray, self.max_features_to_detect) # (N, 1, 2)
            if self.prev_pts is None:
                # 检测失败时返回空结构
                empty_stats = {
                    "timestamp": float(timestamp),
                    "feature_count": 0,
                    "long_track_ratio": 0.0,
                    "mean_parallax": 0.0,
                    "is_stationary": 0,
                    "is_kf_visual": 0,
                    "prev_total_count": 0,
                }
                empty_viz = {
                    "good_prev": np.empty((0, 1, 2)),
                    "good_curr": np.empty((0, 1, 2)),
                    "good_ids": np.array([]),
                }
                return None, None, empty_stats, empty_viz

            # 分配初始ID
            num_new_features = len(self.prev_pts)
            self.prev_pt_ids = np.arange(self.next_feature_id, self.next_feature_id + num_new_features)
            self.next_feature_id += num_new_features

            # 初始化特征点追踪时间
            for feature_id in self.prev_pt_ids:
                self.feature_ages[feature_id] = 1

            if self.use_undistort:
                undistorted_pts = self.undistort_points(self.prev_pts)
            else:
                undistorted_pts = self.prev_pts.reshape(-1, 2) # (N, 2)

            # 第一帧的统计信息
            stats = {
                "timestamp": float(timestamp),
                "feature_count": int(len(self.prev_pt_ids)),
                "long_track_ratio": 0.0,
                "mean_parallax": 0.0,
                "is_stationary": 0,
                "is_kf_visual": 1,
                "prev_total_count": 0,
            }
            viz_payload = {
                "good_prev": self.prev_pts,
                "good_curr": self.prev_pts,
                "good_ids": self.prev_pt_ids,
            }

            # 可视化第一帧（不再在这里调用，由FeatureTracker调用）
            # if self.visualize_flag:
            #     self.visualize_tracking(image, self.prev_pts, self.prev_pts, self.prev_pt_ids, True, False, 0.0, timestamp)

            return undistorted_pts, self.prev_pt_ids, stats, viz_payload

        # 非第一帧处理逻辑       
        # 使用IMU预测的初值（如果提供）
        initial_pts = predicted_pts
        if initial_pts is None and imu_data_for_prediction is not None and imu_processor is not None:
            # 使用GTSAM预积分计算旋转并预测光流初值
            print(f"【VisualProcessor】Using IMU prediction for initial points")
            initial_pts = self._predict_optical_flow_with_imu(imu_data_for_prediction, imu_processor)
        
        # 确保 initial_pts 格式正确（连续数组，形状与 prev_pts 匹配）
        if initial_pts is not None:
            # 确保形状匹配 (N, 1, 2)
            if initial_pts.shape != self.prev_pts.shape:
                initial_pts = initial_pts.reshape(self.prev_pts.shape)
            # 确保是连续的 float32 数组
            initial_pts = np.ascontiguousarray(initial_pts, dtype=np.float32)
        
        # 正向光流
        curr_pts, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, initial_pts,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # 反向光流
        prev_pts_backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self.prev_gray, curr_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        forward_status = forward_status.flatten()

        # 反向光流误差
        # 将数组从 (N, 1, 2) reshape 为 (N, 2) 以便正确计算距离
        prev_pts_reshaped = self.prev_pts.reshape(-1, 2)
        prev_pts_backward_reshaped = prev_pts_backward.reshape(-1, 2)
        fb_error = np.linalg.norm(prev_pts_reshaped - prev_pts_backward_reshaped, axis=1)

        # 最终掩码
        final_mask = (forward_status == 1) & (fb_error < 1.0)

        # 筛选内点
        good_prev = self.prev_pts[final_mask]
        good_curr = curr_pts[final_mask]
        good_ids = self.prev_pt_ids[final_mask]

        # ============== 添加RANSAC几何验证 ==============
        if self.use_ransac and len(good_curr) >= 8:  # 至少需要8个点
            # 去畸变特征点用于本质矩阵估计
            if self.use_undistort:
                good_prev_undist = self.undistort_points(good_prev)
                good_curr_undist = self.undistort_points(good_curr)
            else:
                good_prev_undist = good_prev.reshape(-1, 2) # (N, 2)
                good_curr_undist = good_curr.reshape(-1, 2) # (N, 2)
            
            # 归一化坐标（Essential Matrix需要）
            fx = self.cam_matrix[0, 0]
            fy = self.cam_matrix[1, 1]
            cx = self.cam_matrix[0, 2]
            cy = self.cam_matrix[1, 2]
            
            good_prev_norm = np.column_stack([
                (good_prev_undist[:, 0] - cx) / fx,
                (good_prev_undist[:, 1] - cy) / fy
            ])
            good_curr_norm = np.column_stack([
                (good_curr_undist[:, 0] - cx) / fx,
                (good_curr_undist[:, 1] - cy) / fy
            ])
            
            try:
                # 使用RANSAC估计本质矩阵
                E, ransac_mask = cv2.findEssentialMat(
                    good_prev_norm,
                    good_curr_norm,
                    focal=1.0,  # 已归一化，使用单位焦距
                    pp=(0, 0),  # 已归一化，主点为原点
                    method=cv2.RANSAC,
                    prob=self.ransac_prob,
                    threshold=self.ransac_threshold / fx  # 归一化平面的阈值
                )
                
                if ransac_mask is not None:
                    ransac_mask = ransac_mask.flatten().astype(bool)
                    
                    # 统计RANSAC筛选效果
                    num_before = len(good_curr)
                    num_after = np.sum(ransac_mask)
                    outlier_ratio = 1 - (num_after / num_before)
                    
                    # 只在outlier比例较高时打印警告
                    if outlier_ratio > 0.1:
                        print(f"【RANSAC】Filtered {num_before - num_after}/{num_before} outliers ({outlier_ratio:.1%})")
                    
                    # 应用RANSAC mask
                    good_prev = good_prev[ransac_mask]
                    good_curr = good_curr[ransac_mask]
                    good_ids = good_ids[ransac_mask]
                else:
                    print("【RANSAC Warning】Failed to compute essential matrix, skipping RANSAC filter")
                    
            except cv2.error as e:
                print(f"【RANSAC Error】{e}, skipping RANSAC filter")
        # ===============================================

        # 更新特征点追踪时间
        for feature_id in good_ids:
            if feature_id in self.feature_ages:
                self.feature_ages[feature_id] += 1
            else:
                self.feature_ages[feature_id] = 1

        # 对追踪到的特征点进行空间过滤（按追踪时间优先级）
        good_curr, good_ids, age_mask = self.filter_features_by_age(
            good_curr, good_ids, curr_gray.shape
        )

        # 清理未被保留的特征点年龄记录（应该在空间过滤后）
        tracked_ids_set = set(good_ids)  # 使用空间过滤后的good_ids
        ids_to_remove = [fid for fid in self.feature_ages if fid not in tracked_ids_set]
        for fid in ids_to_remove:
            del self.feature_ages[fid]

        # 同步更新good_prev以匹配过滤后的good_ids
        if len(good_ids) > 0:
            # 创建一个映射，将good_ids映射到good_prev的索引
            prev_id_to_idx = {fid: i for i, fid in enumerate(self.prev_pt_ids[final_mask])}
            # 在good_prev中找到对应good_ids的点
            good_prev_indices = [prev_id_to_idx[fid] for fid in good_ids if fid in prev_id_to_idx]
            good_prev = self.prev_pts[final_mask][good_prev_indices]
        else:
            good_prev = np.empty((0, 1, 2))

        # Measure displacement to mean parallax
        displacement = np.linalg.norm(good_curr.reshape(-1, 2) - good_prev.reshape(-1, 2), axis=1)
        mean_parallax = float(np.mean(displacement)) if len(displacement) > 0 else 0.0

        # is_stationary (0/1)
        is_stationary = int(mean_parallax < self.min_stationary_parallax)

        # long_track_ratio
        long_track_ratio = 0.0
        if len(good_ids) > 0:
            long_track_count = sum(1 for fid in good_ids if self.feature_ages.get(fid, 0) >= self.long_track_age_threshold)
            long_track_ratio = float(long_track_count / len(good_ids))

        # 判断关键帧视觉条件（is_kf_visual）
        is_kf_visual = 0
        # 条件1：平均视差大于最小视差
        if mean_parallax > self.min_parallax:
            is_kf_visual = 1
        # 条件2：长追踪点比例小于最小长追踪比例
        if len(good_ids) > 0 and long_track_ratio < self.min_long_track_ratio:
            is_kf_visual = 1

        num_current_features = len(good_curr)

        final_pts = good_curr
        final_ids = good_ids
        
        # 补充特征点
        if len(good_curr) < self.max_features_to_detect:
            # 判断关键视觉条件3：
            # 跟踪到的特征点数量小于最小跟踪比例（新特征点数量大于一定比例），则认为是关键帧
            if len(good_curr) < (self.max_features_to_detect * self.min_track_ratio):
                is_kf_visual = 1

            # 使用重过滤生成的mask来检测新特征点
            num_new_features_needed = self.max_features_to_detect - num_current_features
            new_pts = self.detect_features(curr_gray, num_new_features_needed, mask=age_mask)

            if new_pts is not None:
                # 分配新的特征点id
                num_new = len(new_pts)
                new_ids = np.arange(self.next_feature_id, self.next_feature_id + num_new)
                self.next_feature_id += num_new

                # 初始化新特征点的年龄为1
                for new_id in new_ids:
                    self.feature_ages[new_id] = 1

                final_pts = np.vstack([good_curr, new_pts])
                final_ids = np.hstack([good_ids, new_ids])
        
        self.prev_gray = curr_gray
        self.prev_pts = final_pts
        self.prev_pt_ids = final_ids

        # 特征点去畸变
        if self.use_undistort:
            undistorted_final_pts = self.undistort_points(final_pts)
        else:
            undistorted_final_pts = final_pts.reshape(-1, 2) # (N, 2)

        # 准备返回的统计信息和可视化数据
        stats = {
            "timestamp": float(timestamp),
            "feature_count": int(len(good_ids)),
            "long_track_ratio": float(long_track_ratio),
            "mean_parallax": float(mean_parallax),
            "is_stationary": int(is_stationary),
            "is_kf_visual": int(is_kf_visual),
            "prev_total_count": int(prev_total_count),
        }
        viz_payload = {
            "good_prev": good_prev,
            "good_curr": good_curr,
            "good_ids": good_ids,
        }

        return undistorted_final_pts, final_ids, stats, viz_payload

    def get_age_color(self, age):
        """直接RGB插值"""
        normalized_age = min(age / self.max_age_for_color, 1.0)
        
        # 红(255,0,0) -> 蓝(0,0,255)
        r = int(255 * (1 - normalized_age))
        b = int(255 * normalized_age)
        g = 0
        
        return (b, g, r)  # BGR格式
    
    def visualize_tracking(self, image, good_prev, good_curr, good_ids, is_kf, is_stationary, mean_parallax, timestamp, prev_total_count=0, long_track_ratio=0.0):
        """
        可视化特征点追踪结果
        
        Returns:
            vis_img: 可视化后的图像（numpy数组）
        """
        vis_img = image.copy()
        
        # 绘制每个特征点的轨迹和信息
        for p1, p2, feature_id in zip(good_prev, good_curr, good_ids):
            p1_t = tuple(p1.ravel().astype(int))
            p2_t = tuple(p2.ravel().astype(int))
            
            # 获取该特征点的年龄
            age = self.feature_ages.get(feature_id, 0)
            
            # 根据年龄获取渐变颜色
            color = self.get_age_color(age)
            
            # 画出光流轨迹（使用渐变颜色）
            cv2.arrowedLine(vis_img, p1_t, p2_t, color=[0, 255, 0], thickness=2, tipLength=0.3)
            
            # 画出特征点（使用相同的渐变颜色）
            cv2.circle(vis_img, p2_t, 4, color, -1)
            
            # 画出特征点的ID和年龄（白底黑字，更清晰）
            label = f"{feature_id}"
            cv2.putText(vis_img, label, (p2_t[0]+5, p2_t[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # 显示统计信息（不再写入日志，日志由FeatureTracker负责）
        if len(good_ids) > 0:
            # 计算追踪成功率
            if prev_total_count > 0:
                tracking_rate = len(good_ids) / prev_total_count
                info_text1 = (f"Features: {len(good_ids)}/{prev_total_count} ({tracking_rate:.0%}) | "
                             f"KF: {is_kf} | "
                             f"Stationary: {is_stationary} | Parallax: {mean_parallax:.2f}")
            else:
                info_text1 = f"Features: {len(good_ids)} | Long: {long_track_ratio:.0%} | KF: {is_kf} | Stationary: {is_stationary} | Parallax: {mean_parallax:.2f}"
        else:
            info_text1 = f"Features: 0 | KF: {is_kf}"

        cv2.putText(vis_img, info_text1, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow("Optical Flow", vis_img)
        cv2.waitKey(1)
        
        return vis_img
    
    def _predict_optical_flow_with_imu(self, imu_data_for_prediction, imu_processor):
        """
        使用GTSAM预积分的旋转部分预测光流初值
        
        Args:
            imu_data_for_prediction: 包含IMU测量数据和时间戳的字典
                - 'measurements': List of (timestamp, ImuMeasurement) tuples
                - 'start_time': 起始时间戳
                - 'end_time': 结束时间戳
            imu_processor: IMUProcessor实例，用于执行预积分
            
        Returns:
            predicted_pts: 预测的特征点位置 (N, 1, 2) 或 None
        """
        if imu_data_for_prediction is None or imu_processor is None:
            return None
        
        # 获取上一帧的特征点
        if self.prev_pts is None or len(self.prev_pts) == 0:
            return None
        
        measurements = imu_data_for_prediction['measurements']
        start_time = imu_data_for_prediction['start_time']
        end_time = imu_data_for_prediction['end_time']
        
        if len(measurements) < 1:
            return None
        
        # 使用GTSAM预积分API计算旋转增量
        try:
            preintegrated = imu_processor.pre_integration(measurements, start_time, end_time)
            if preintegrated is None:
                return None
            
            # 从预积分结果中获取旋转增量（deltaRij返回从i到j的旋转）
            delta_R_imu = preintegrated.deltaRij().matrix()  # 返回numpy数组，形状(3, 3)
            
        except Exception as e:
            print(f"【VisualProcessor】Error in IMU preintegration: {e}")
            return None
        
        # 将特征点从像素坐标转换为归一化坐标
        # prev_pts shape: (N, 1, 2)
        prev_pts_reshaped = self.prev_pts.reshape(-1, 2)  # (N, 2)
        
        # 去畸变（如果需要）
        # 这里P=None，返回归一化平面坐标
        if self.use_undistort:
            prev_pts_normalized = cv2.undistortPoints(
                self.prev_pts, self.cam_matrix, self.dist_coeffs,
                P=None
            ).reshape(-1, 2)
        else:
            # 直接归一化
            prev_pts_homogeneous = np.hstack([prev_pts_reshaped, np.ones((len(prev_pts_reshaped), 1))])
            prev_pts_normalized = (np.linalg.inv(self.cam_matrix) @ prev_pts_homogeneous.T).T[:, :2]
        
        # 转换为齐次坐标 (N, 3)
        prev_pts_homogeneous = np.hstack([prev_pts_normalized, np.ones((len(prev_pts_normalized), 1))])
        
        # 从IMU坐标系旋转转换为相机坐标系旋转
        # 获取T_bc外参
        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        T_bc = np.asarray(T_bc_raw).reshape(4, 4)
        R_bc = T_bc[:3, :3]
        
        # 从IMU坐标系旋转转换为相机坐标系旋转
        # R_cam_curr_cam_prev = R_bc @ R_imu_curr_imu_prev @ R_bc^T
        R_cam_curr_cam_prev = R_bc.T @ delta_R_imu @ R_bc
        
        # 应用旋转（在归一化坐标系中）
        curr_pts_normalized = (R_cam_curr_cam_prev @ prev_pts_homogeneous.T).T[:, :2]
        
        # 转换回像素坐标
        curr_pts_homogeneous = np.hstack([curr_pts_normalized, np.ones((len(curr_pts_normalized), 1))])
        curr_pts_pixel = (self.cam_matrix @ curr_pts_homogeneous.T).T[:, :2]
        
        # 转换回 (N, 1, 2) 格式，确保是连续的数组
        predicted_pts = np.ascontiguousarray(
            curr_pts_pixel.reshape(-1, 1, 2).astype(np.float32)
        )
        
        return predicted_pts