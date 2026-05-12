from hmac import new
import queue
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B, L
# from gtsam_unstable import IncrementalFixedLagSmoother, FixedLagSmootherKeyTimestampMap
from gtsam import IncrementalFixedLagSmoother

import re
from utils.debug import Debugger
import time

class Backend:
    def __init__(self, global_central_map, config, imu_processor):
        self.global_central_map = global_central_map
        self.config = config

        # 使用 iSAM2 作为优化器
        # self.lag_window_size = config.get('lag_window_size', 9) # 优化器的滑窗
        # parameters = gtsam.ISAM2Params()
        # parameters.setRelinearizeThreshold(0.01) 
        # parameters.relinearizeSkip = 1
        # self.smoother = IncrementalFixedLagSmoother(self.lag_window_size, parameters) # 自动边缘化

        # 滑窗数据结构
        self.active_values = gtsam.Values()
        self.gtsam_kf_gtsam_ids = []
        self.marg_factor = None
        
        # 鲁棒因子
        self.visual_noise_sigma = config.get('visual_noise_sigma', 2.0)
        self.visual_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.visual_noise_sigma)
        self.visual_robust_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), self.visual_noise)

        # 是否使用深度降权
        self.use_depth_weight = config.get('use_depth_weight', False)
        # 添加深度降权参数
        self.depth_weight_base = config.get('depth_weight_base', 5.0)  # 基础深度阈值（米）
        self.depth_weight_max = config.get('depth_weight_max', 3.0)  # 最大噪声倍数
        self.depth_weight_power = config.get('depth_weight_power', 1.5)  # 深度权重指数
        self.new_landmark_inflation_ratio = config.get('new_landmark_inflation_ratio', 5.0)

        # 预优化最大重投影误差
        self.rejection_threshold = config.get('rejection_threshold', 400.0)

        # 状态与id管理
        self.kf_id_to_gtsam_id = {}
        self.landmark_id_to_gtsam_id = {}
        self.next_gtsam_kf_id = 0
        self.factor_indices_to_remove = []

        # 获取相机内、外参
        cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)
        self.K = gtsam.Cal3_S2(cam_intrinsics[0, 0], cam_intrinsics[1, 1], 0, 
                               cam_intrinsics[0, 2], cam_intrinsics[1, 2])

        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)
        self.body_T_cam = gtsam.Pose3(self.T_bc)
        self.cam_T_body = self.body_T_cam.inverse()

        # 存储最新的优化后的偏置，用于IMU预积分
        self.latest_bias = gtsam.imuBias.ConstantBias()

        # 定义要记录的列
        log_columns = [
            "gtsam_id", "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "bias_acc_x", "bias_acc_y", "bias_acc_z",
            "bias_gyro_x", "bias_gyro_y", "bias_gyro_z",
            "new_factors_error"
        ]
        # 初始化Debugger
        self.logger = Debugger(self.config, file_prefix="backend_state", column_names=log_columns)

    # 关键帧id映射到图的id
    def _get_kf_gtsam_id(self, kf_id):
        if kf_id not in self.kf_id_to_gtsam_id:
            self.kf_id_to_gtsam_id[kf_id] = self.next_gtsam_kf_id
            self.next_gtsam_kf_id += 1
        return self.kf_id_to_gtsam_id[kf_id]

    # 路标点id映射到图的id
    def _get_lm_gtsam_id(self, lm_id):
        if lm_id not in self.landmark_id_to_gtsam_id:
            self.landmark_id_to_gtsam_id[lm_id] = lm_id
        return self.landmark_id_to_gtsam_id[lm_id]

    def get_latest_optimized_state(self):
        if self.next_gtsam_kf_id == 0:
            return None, None, None
        
        latest_gtsam_id = self.next_gtsam_kf_id - 1

        result = self.smoother.calculateEstimate()

        try:
            pose = result.atPose3(X(latest_gtsam_id))
            velocity = result.atVector(V(latest_gtsam_id))
            bias = result.atConstantBias(B(latest_gtsam_id))
            # print(f"【Backend】: Latest optimized state: pose: {pose.matrix()}, velocity: {velocity}, bias: {bias}")
            return pose, velocity, bias
        except Exception as e:
            print(f"[Error][Backend] Failed to retrieve latest state for gtsam_id {latest_gtsam_id}: {e}")
            return None, None, None

    def update_estimator_map(self, keyframe_window, landmarks):
        print("【Backend】: Syncing optimized results back to Estimator...")
        optimized_results = self.smoother.calculateEstimate()

        # 更新关键帧位姿
        for kf in keyframe_window:
           # 获取待更新关键帧的gtsam_id
            gtsam_id = self.kf_id_to_gtsam_id.get(kf.get_id())
            if gtsam_id is not None and optimized_results.exists(X(gtsam_id)):
                
                # 从优化结果中获取最新的IMU位姿 T_w_b并更新
                pose_w_b = optimized_results.atPose3(X(gtsam_id))
                kf.set_global_pose(pose_w_b.matrix())

        # 更新路标点坐标
        for lm_id, landmark_obj in landmarks.items():
            gtsam_id = self._get_lm_gtsam_id(lm_id)
            if gtsam_id is not None and optimized_results.exists(L(gtsam_id)):
                # 1. 从优化结果中获取最新的3D坐标
                optimized_position = optimized_results.atPoint3(L(gtsam_id))
                # 2. 调用对象的方法来更新其内部状态
                landmark_obj.set_triangulated(optimized_position)
                # print(f"【Backend】: Updated landmark {lm_id} to {optimized_position}")

    def remove_stale_landmarks(self, unhealty_lm_ids, unhealty_lm_ids_depth, 
                                unhealty_lm_ids_reproj, oldest_kf_id_in_window):
        print(f"【Backend】: 接收到移除 {len(unhealty_lm_ids)} 个陈旧路标点的指令。")
        if not unhealty_lm_ids:
            return

        # 不再手动删除因子！
        # 原因：手动删除因子会与Fixed-Lag Smoother的自动边缘化机制冲突
        # 导致 IndexError: map::at
        
        # 只删除ID映射，阻止这些landmark再次被添加到图中
        for lm_id in unhealty_lm_ids:
            if lm_id in self.landmark_id_to_gtsam_id:
                del self.landmark_id_to_gtsam_id[lm_id]
                print(f"【Backend】: 已移除 landmark {lm_id} 的ID映射")

        print(f"【Backend】: 成功标记 {len(unhealty_lm_ids)} 个路标点为待清理状态")
        print(f"【Backend】: Fixed-Lag Smoother 将在滑窗移动时自动清理这些landmark")
        

    def initialize_optimize(self, initial_keyframes, initial_imu_factors, initial_landmarks, initial_velocities, initial_bias):
        print("【Backend】: Initializing optimize...")

        graph = gtsam.NonlinearFactorGraph()
        estimates = gtsam.Values()
        
        # initial_window_stamps = FixedLagSmootherKeyTimestampMap()
        initial_window_stamps = {}

        for i, kf in enumerate(initial_keyframes):
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())

            # 从初始化结果中获取位姿、速度和偏置
            T_wb = gtsam.Pose3(kf.get_global_pose())
            # initial_velocities 是一个扁平化的数组，每3个元素是一个速度向量
            velocity = initial_velocities[i*3 : i*3+3]
            
            # 所有帧使用相同的初始偏置
            bias = initial_bias

            # 添加初始估计值
            estimates.insert(X(kf_gtsam_id), T_wb)
            estimates.insert(V(kf_gtsam_id), velocity)
            estimates.insert(B(kf_gtsam_id), bias)

            # 添加滑窗记录
            initial_window_stamps[X(kf_gtsam_id)] = float(kf_gtsam_id)
            initial_window_stamps[V(kf_gtsam_id)] = float(kf_gtsam_id)
            initial_window_stamps[B(kf_gtsam_id)] = float(kf_gtsam_id)

            # 为第一帧添加强先验
            if kf_gtsam_id == 0:
                prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*3 + [1e-2]*3))
                prior_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2e-2] * 3))
                prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1]*3 + [1e-2]*3))
                graph.add(gtsam.PriorFactorPose3(X(0), T_wb, prior_pose_noise))
                graph.add(gtsam.PriorFactorVector(V(0), velocity, prior_vel_noise))
                graph.add(gtsam.PriorFactorConstantBias(B(0), bias, prior_bias_noise))
        
        # 为每一个landmark设置滑窗记录
        last_gtsam_id = self._get_kf_gtsam_id(initial_keyframes[-1].get_id())
        for lm_id in initial_landmarks.keys():
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            initial_window_stamps[L(lm_gtsam_id)] = float(last_gtsam_id) # 设为最后一帧的ID

        # 添加所有初始IMU因子
        for factor_data in initial_imu_factors:
            start_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['start_kf_timestamp'])
            end_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['end_kf_timestamp'])
            gtsam_id1 = self._get_kf_gtsam_id(start_kf.get_id())
            gtsam_id2 = self._get_kf_gtsam_id(end_kf.get_id())
            pim = factor_data['imu_preintegration']
            graph.add(gtsam.CombinedImuFactor(X(gtsam_id1), V(gtsam_id1), X(gtsam_id2), V(gtsam_id2), B(gtsam_id1), B(gtsam_id2), pim))

        # 添加所有初始路标点变量和视觉因子
        for lm_id, lm_3d_pos in initial_landmarks.items():
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            estimates.insert(L(lm_gtsam_id), lm_3d_pos)

        for kf in initial_keyframes:
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                # 只处理本次优化中新添加的landmark
                if lm_id in initial_landmarks:
                    lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
                    # 计算深度并应用降权
                    T_wb = gtsam.Pose3(kf.get_global_pose()) # 获取关键帧位姿用于深度计算
                    current_lm_pos = initial_landmarks[lm_id]
                    depth = self._compute_landmark_depth(current_lm_pos, T_wb)
                    # 这里将初始化的点标记为False
                    weighted_noise = self._get_adaptive_noise(depth, False)

                    factor = gtsam.GenericProjectionFactorCal3_S2(
                        pt_2d, weighted_noise, X(kf_gtsam_id), L(lm_gtsam_id), 
                        self.K, body_P_sensor=self.body_T_cam
                    )
                    graph.add(factor)

        # 执行iSAM2的第一次更新（批量模式）
        print(f"【Backend】: Initializing iSAM2 with {graph.size()} new factors and {estimates.size()} new values...")
        
        try:
            start_time = time.time()
            self.smoother.update(graph, estimates, initial_window_stamps)
            end_time = time.time()
            print(f"【Backend Timer】: Initial optimization took { (end_time - start_time) * 1000:.3f} ms.")
        except RuntimeError as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!! INITIALIZATION FAILED !!!!!!!!!!!!!!")
            print(f"ERROR: {e}")
            return # 失败时必须返回

        # 更新最新bias
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        print(f"【Backend】: Latest optimized state: pose: {latest_pose.matrix()}, velocity: {latest_vel}, bias: {latest_bias}")

        latest_gtsam_id = self.next_gtsam_kf_id - 1
        print(f"【Backend】: Latest gtsam_id: {latest_gtsam_id}")
        if latest_bias is not None:
            self.latest_bias = latest_bias
        print("【Backend】: Initial graph optimization complete.")

        # 记录优化状态
        new_factors_error = self._log_optimization_error(graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)


    def optimize_incremental(self, last_keyframe, new_keyframe, new_imu_factors, 
                            new_landmarks, new_visual_factors, initial_state_guess, is_stationary, oldest_kf_id_in_window):
        new_graph = gtsam.NonlinearFactorGraph()
        new_estimates = gtsam.Values()
        current_isam_values = self.smoother.calculateEstimate()
        new_window_stamps = {}

        # 添加新关键帧的状态变量，使用IMU预测值作为初始估计
        kf_gtsam_id = self._get_kf_gtsam_id(new_keyframe.get_id())
        T_wb_guess, vel_guess, bias_guess = initial_state_guess

        # 检查关键帧是否已经在图中存在，避免重复添加（防御性检查）
        if not current_isam_values.exists(X(kf_gtsam_id)) or not current_isam_values.exists(V(kf_gtsam_id)) or not current_isam_values.exists(B(kf_gtsam_id)):
            new_estimates.insert(X(kf_gtsam_id), T_wb_guess)
            new_estimates.insert(V(kf_gtsam_id), vel_guess)
            new_estimates.insert(B(kf_gtsam_id), bias_guess)

            # 添加滑窗记录
            new_window_stamps[X(kf_gtsam_id)] = float(kf_gtsam_id)
            new_window_stamps[V(kf_gtsam_id)] = float(kf_gtsam_id)
            new_window_stamps[B(kf_gtsam_id)] = float(kf_gtsam_id)
        else:
            print(f"【Backend】: Warning: Keyframe {new_keyframe.get_id()} (gtsam_id={kf_gtsam_id}) already exists in graph. Skipping variable insertion.")
            # 如果关键帧已存在，仍然需要更新滑窗时间戳（如果Fixed-Lag Smoother需要）
            # 注意：这里不添加变量，只更新时间戳（如果需要的话）

        # if not is_stationary:
        # 添加IMU因子
        last_kf_gtsam_id = self._get_kf_gtsam_id(last_keyframe.get_id())
        pim = new_imu_factors['imu_preintegration']
        imu_factor = gtsam.CombinedImuFactor(
            X(last_kf_gtsam_id), V(last_kf_gtsam_id), X(kf_gtsam_id), V(kf_gtsam_id),
            B(last_kf_gtsam_id), B(kf_gtsam_id), pim)
        new_graph.add(imu_factor)

        # 添加新路标点顶点，注意这里添加的顶点只在new_estimates中还没有进入isam2的图
        # if not is_stationary:
        added_new_landmark_gtsam_ids = set()

        for lm_id, lm_3d_pos in new_landmarks.items():            
            # 增加一个NaN/Inf的显式检查，这对于调试崩溃至关重要
            if np.isnan(lm_3d_pos).any() or np.isinf(lm_3d_pos).any():
                print(f"🔥 【Backend】[致命警告]: 路标点 L{lm_id} 的初始值无效 (NaN/Inf)！优化即将因此崩溃！")
                continue  # 直接跳过无效的landmark

            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            # ---!!!--- 在此处添加您要的日志 ---!!!---
            # 打印即将送入优化器的路标点的值
            # print(f"🕵️‍ 【Backend】: 优化器即将处理新路标点 L{lm_id}，其三角化初始值为: {lm_3d_pos}")

            # 检查：1) 不在旧图中，2) 还没被添加过 确保顶点只被添加一次
            if not current_isam_values.exists(L(lm_gtsam_id)):
                new_estimates.insert(L(lm_gtsam_id), lm_3d_pos)
                # 添加新路标点的滑窗记录
                new_window_stamps[L(lm_gtsam_id)] = float(kf_gtsam_id)
                added_new_landmark_gtsam_ids.add(lm_gtsam_id)
        
        # 如果一个新路标点在 estimates 里，但所有因子都被 chi2 拒绝，必须将其从 estimates 移除
        # 否则会导致 iSAM2 遇到无约束变量而奇异/崩溃
        valid_new_landmarks = set()

        # -------------------------------------------------------------------------
        # 【新增逻辑】: 因子防火墙 (Factor Firewall)
        # -------------------------------------------------------------------------
        
        valid_visual_factors = []
        bad_landmarks = set() # 记录坏点

        for kf_id, lm_id, pt_2d in new_visual_factors:
            # 1. 基础检查
            if lm_id not in self.landmark_id_to_gtsam_id:
                continue
            
            kf_gtsam_id = self._get_kf_gtsam_id(kf_id)
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)

            # 2. 准备计算误差所需的临时变量
            # 我们需要获取 kf 的位姿 和 lm 的位置
            # 情况A: 变量在 new_estimates 中 (本帧新加的)
            # 情况B: 变量在 current_isam_values 中 (老变量)
            
            pose = None
            if new_estimates.exists(X(kf_gtsam_id)):
                pose = new_estimates.atPose3(X(kf_gtsam_id))
            elif current_isam_values.exists(X(kf_gtsam_id)):
                pose = current_isam_values.atPose3(X(kf_gtsam_id))
            
            point = None
            is_new_point = False
            if new_estimates.exists(L(lm_gtsam_id)):
                point = new_estimates.atPoint3(L(lm_gtsam_id))
                is_new_point = True
            elif current_isam_values.exists(L(lm_gtsam_id)):
                point = current_isam_values.atPoint3(L(lm_gtsam_id))
            
            # 如果我们找不到位姿或点，就没法计算误差，只能先跳过 (或保守添加)
            if pose is None or point is None:
                continue

            # 3. 构造临时因子计算误差
            # 注意：这里我们还没真的加到 new_graph，只是模拟一下
            if self.use_depth_weight:
                depth = self._compute_landmark_depth(point, pose)
            else:
                depth = None
            
            # 使用严格的噪声模型进行检测 (不加 Hubber，看原始误差)
            check_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0) 
            temp_factor = gtsam.GenericProjectionFactorCal3_S2(
                pt_2d, check_noise, X(kf_gtsam_id), L(lm_gtsam_id), 
                self.K, body_P_sensor=self.body_T_cam
            )
            
            # 构造临时 Values
            temp_values = gtsam.Values()
            temp_values.insert(X(kf_gtsam_id), pose)
            temp_values.insert(L(lm_gtsam_id), point)
            
            try:
                # 计算未经鲁棒核抑制的原始像素误差
                error = temp_factor.error(temp_values)
            except:
                error = float('inf')

            # 4. 判决时刻！
            # 阈值设定：
            # Error = 0.5 * (u-u')^2 / sigma^2
            # 如果 sigma=1, error=50 意味着像素误差 sqrt(100) = 10 像素
            # error=1618 意味着像素误差极极大
            
            REJECTION_THRESHOLD = self.rejection_threshold  # 对应约 10 像素的重投影误差
            
            if error > REJECTION_THRESHOLD:
                # 这是一个坏因子！
                print(f"🔥 [Firewall] 拦截坏因子! KF{kf_id}-LM{lm_id}, Error: {error:.2f}")
                bad_landmarks.add(lm_id) # 标记这个点有问题
                
                # 如果这是一个老点 (不在 new_estimates 里)，它可能已经腐化了
                # 我们不仅要拒绝这个因子，甚至应该考虑把这个点拉黑
            else:
                # 通过检查，加入待添加列表
                valid_visual_factors.append((kf_id, lm_id, pt_2d, depth, is_new_point))

        # -------------------------------------------------------------------------
        # 正式添加通过检查的因子到 new_graph
        # -------------------------------------------------------------------------
        
        for kf_id, lm_id, pt_2d, depth, is_new_point in valid_visual_factors:
            # 如果这个点已经被标记为坏点（因为在别的帧视角下误差巨大），那么它的所有因子都不要了
            if lm_id in bad_landmarks:
                continue
                
            kf_gtsam_id = self._get_kf_gtsam_id(kf_id)
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            
            # ... (这里放你原本的构建 factor 代码，使用 Huber 核等) ...
            # weighted_noise = self._get_adaptive_noise(depth, is_new_point)
            factor = gtsam.GenericProjectionFactorCal3_S2(
                    pt_2d, self.visual_robust_noise, X(kf_gtsam_id), L(lm_gtsam_id), 
                    self.K, body_P_sensor=self.body_T_cam
                )
            new_graph.add(factor)
            
            # 更新时间戳逻辑...
            if not is_new_point: # old_lm_exists
                 new_window_stamps[L(lm_gtsam_id)] = float(self._get_kf_gtsam_id(new_keyframe.get_id()))
            elif lm_id not in bad_landmarks:
                 valid_new_landmarks.add(lm_gtsam_id) # 这是一个有效的新点

        # 清理垃圾：把刚才发现的 bad_landmarks 从 new_estimates 里删掉
        # 防止把没有因子的孤立点加进去，导致 Indeterminant
        for lm_id in bad_landmarks:
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            if new_estimates.exists(L(lm_gtsam_id)):
                print(f"🗑️ [Firewall] 移除有毒的新点变量 L{lm_id}")
                new_estimates.erase(L(lm_gtsam_id))
            if L(lm_gtsam_id) in new_window_stamps:
                del new_window_stamps[L(lm_gtsam_id)]
        
        # 清理无效的新路标点
        # 遍历本次尝试添加的所有新路标点
        for lm_id in list(new_landmarks.keys()): 
            if lm_id not in self.landmark_id_to_gtsam_id: continue
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)

            # 如果它在 estimates 里（说明通过了 NaN 检查），但不在 valid 集合里（说明没因子）
            if new_estimates.exists(L(lm_gtsam_id)) and lm_gtsam_id not in valid_new_landmarks:
                # print(f"【Backend】: Cleaning up unconstrained new landmark L{lm_id} (All factors rejected)")
                new_estimates.erase(L(lm_gtsam_id))
                if L(lm_gtsam_id) in new_window_stamps:
                    del new_window_stamps[L(lm_gtsam_id)]

        #     print(f"【Backend】: Added {len(new_landmarks)} new landmarks and {len(new_visual_factors)} visual factors.")
        # else:
        #     print("【Backend】: Skipped visual landmarks and factors due to stationary state.")

        # ======================= ZERO-VELOCITY UPDATE (ZUPT) & NO-MOTION POSE FACTOR =======================
        if is_stationary:
            # 添加零速度更新因子
            last_kf_gtsam_id = self._get_kf_gtsam_id(last_keyframe.get_id())
            kf_gtsam_id = self._get_kf_gtsam_id(new_keyframe.get_id())
            zero_velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.03)
            zero_velocity_prior = gtsam.PriorFactorVector(V(kf_gtsam_id), np.zeros(3), zero_velocity_noise)
            new_graph.add(zero_velocity_prior)
            print("【Backend】: Added Zero-Velocity-Update (ZUPT) factor.")

            # # 添加单位位姿因子
            # no_motion_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            #     np.array([0.01, 0.01, 0.01,  # 旋转轴 (roll, pitch, yaw)
            #               0.03, 0.03, 0.03])) # 平移 (x, y, z)
            
            # new_graph.add(gtsam.BetweenFactorPose3(X(last_kf_gtsam_id), X(kf_gtsam_id),      
            #               gtsam.Pose3(), no_motion_pose_noise))
            # print("【Backend】: Added No-Motion Pose Factor.")
        # ============================================================================================

        # 执行iSAM2增量更新
        # graph = self.smoother.getFactors()
        # print("【Backend】: graph: ", graph)
        # print(f"【Backend】: Updating iSAM2 ({new_graph.size()} new factors, {new_estimates.size()} new variables)...")
        
        try:
            start_time = time.time()
            self.smoother.update(new_graph, new_estimates, new_window_stamps)
            end_time = time.time()
            print(f"【Backend Timer】: Incremental optimization took { (end_time - start_time) * 1000:.3f} ms.")

        # except Exception as e:
        except RuntimeError as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!! OPTIMIZATION FAILED !!!!!!!!!!!!!!")
            print(f"ERROR: {e}")
            return

        # 更新最新bias
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        print(f"【Backend】: Latest optimized state: pose: {latest_pose.matrix()}, velocity: {latest_vel}, bias: {latest_bias}")
        latest_gtsam_id = self.next_gtsam_kf_id - 1
        if latest_bias is not None:
            self.latest_bias = latest_bias

        if latest_pose is None:
             print("【Backend】Critical: Optimization succeeded but state retrieval failed.")
             return

        # 记录优化误差
        new_factors_error = self._log_optimization_error(new_graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        print("【Backend】: Incremental optimization complete.")


    def _log_optimization_error(self, new_factors_graph):
        try:
            optimized_result = self.smoother.calculateEstimate()
            new_factors_error = new_factors_graph.error(optimized_result)

            current_full_graph = self.smoother.getFactors()

            print(f"【Backend】优化误差统计: "
                  f"本轮新增因子误差 = {new_factors_error:.4f}")

            # ======================= DETAILED FACTOR ERROR LOGGING =======================
            debug_start_frame = 0 # 设为0以立即开始打印
            latest_gtsam_id = self.next_gtsam_kf_id - 1
            if latest_gtsam_id >= debug_start_frame:
                print("\n" + "="*40 + f" DETAILED ERROR ANALYSIS (Frame {latest_gtsam_id}) " + "="*40)
                
                # 遍历图中的所有因子
                for i in range(current_full_graph.size()):
                    factor = current_full_graph.at(i)
                    if factor is None: # 检查因子是否有效
                        continue
                        
                    try:
                        # 计算这个特定因子的误差
                        error = factor.error(optimized_result)
                        
                        # 打印误差大于阈值的因子，以避免日志刷屏
                        if error > 10.0: 
                            # 打印因子的Python类名
                            factor_type = factor.__class__.__name__
                            print(f"  - Factor {i}: Error = {error:.4f}, Type = {factor_type}")
                            
                            # 尝试打印与该因子相关的Key
                            keys = factor.keys()
                            key_str = ", ".join([gtsam.DefaultKeyFormatter(key) for key in keys])
                            print(f"    Keys: [{key_str}]")
                            
                    except Exception as e_factor:
                        # 捕获计算单个因子误差时可能发生的错误
                        print(f"  - Factor {i}: 无法计算误差或获取Keys. Error: {e_factor}")

                print("="*100 + "\n")
            # ===========================================================================
            
            return new_factors_error
            
        except Exception as e:
            print(f"[Error][Backend] 计算优化误差时出错: {e}")
            return -1.0, -1.0
        
    def _log_state_and_errors(self, latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error):
        position = latest_pose.translation()
        acc_bias = latest_bias.accelerometer()
        gyro_bias = latest_bias.gyroscope()

        state_data = {
            "gtsam_id": latest_gtsam_id,
            "pos_x": position[0], "pos_y": position[1], "pos_z": position[2],
            "vel_x": latest_vel[0], "vel_y": latest_vel[1], "vel_z": latest_vel[2],
            "bias_acc_x": acc_bias[0], "bias_acc_y": acc_bias[1], "bias_acc_z": acc_bias[2],
            "bias_gyro_x": gyro_bias[0], "bias_gyro_y": gyro_bias[1], "bias_gyro_z": gyro_bias[2],
            "new_factors_error": new_factors_error
        }
        self.logger.log_state(state_data)

    def _get_adaptive_noise(self, depth, is_new_landmark):
        """
        结合深度权重和新点膨胀的自适应噪声模型
        depth: landmark到相机的深度（米）
        is_new_landmark: 是否为刚入图的新点
        """
        # 1. 第一层：计算基于深度的基础噪声 (Base Sigma)
        if depth is None:
            base_sigma = 2.0
        else:
            if depth <= self.depth_weight_base:
                base_sigma = 2.0 # 基础像素噪声
            else:
                # 深度越远，噪声越大
                depth_ratio = depth / self.depth_weight_base
                # 限制一下最大深度倍数，防止无穷远点导致数值问题
                clamped_ratio = min(depth_ratio, 5.0) 
                weight_factor = 1.0 + (clamped_ratio ** self.depth_weight_power) * (self.depth_weight_max - 1.0)
                base_sigma = 2.0 * weight_factor
        
        # 2. 第二层：如果是新点，应用膨胀系数 (Inflation)
        if is_new_landmark:
            final_sigma = base_sigma * self.new_landmark_inflation_ratio
        else:
            final_sigma = base_sigma
            
        # 3. 创建 Huber 鲁棒核噪声模型
        noise_model = gtsam.noiseModel.Isotropic.Sigma(2, final_sigma)
        robust_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(2.5), 
            noise_model
        )
        return robust_noise

    def _compute_landmark_depth(self, lm_3d_pos, kf_pose):
        # 获取body到相机的变换
        T_bc = self.T_bc
        R_bc = T_bc[:3, :3]
        t_bc = T_bc[:3, 3]
        
        # 计算相机在世界坐标系下的位置
        T_w_b = kf_pose.matrix()
        R_w_b = T_w_b[:3, :3]
        t_w_b = T_w_b[:3, 3]
        
        # 相机位置 = body位置 + R_w_b @ t_bc
        cam_pos_w = t_w_b + R_w_b @ t_bc
        
        # 计算深度（世界坐标系下的距离）
        # 修复：使用 try-except 而不是 isinstance，因为 gtsam.Point3 可能不可直接访问
        try:
            # 尝试使用 x(), y(), z() 方法（GTSAM Point3对象）
            lm_pos_w = np.array([lm_3d_pos.x(), lm_3d_pos.y(), lm_3d_pos.z()])
        except AttributeError:
            # 如果不是Point3对象，直接转换为numpy数组
            lm_pos_w = np.array(lm_3d_pos)
        
        depth = np.linalg.norm(lm_pos_w - cam_pos_w)
        return depth
