import time
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B, L

from utils.debug import Debugger
from datatype.landmark import LandmarkStatus

class Backend:
    def __init__(self, global_central_map, config, imu_processor):
        self.global_central_map = global_central_map
        self.config = config

        # 使用 iSAM2 作为优化器
        # parameters = gtsam.ISAM2Params()
        # parameters.setRelinearizeThreshold(0.01) 
        # parameters.relinearizeSkip = 1
        # self.smoother = IncrementalFixedLagSmoother(self.lag_window_size, parameters) # 自动边缘化

        # 滑窗数据结构
        self.lag_window_size = config.get('lag_window_size', 9) # 优化器的滑窗
        self.active_values = gtsam.Values()
        self.active_kf_gtsam_ids = []
        self.marg_factor = None

        # 初始化先验因子缓存
        self.init_priors = gtsam.NonlinearFactorGraph()

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

        # 🔥 新增：用于彻底拦截已经被化为先验的路标点，防止前端诈尸
        self.marginalized_landmarks = set()

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
        if not self.active_kf_gtsam_ids:
            return None, None, None
        
        latest_gtsam_id = self.active_kf_gtsam_ids[-1]

        try:
            pose = self.active_values.atPose3(X(latest_gtsam_id))
            velocity = self.active_values.atVector(V(latest_gtsam_id))
            bias = self.active_values.atConstantBias(B(latest_gtsam_id))
            return pose, velocity, bias
        except Exception as e:
            print(f"[Error][Backend] Failed to retrieve latest state: {e}")
            return None, None, None

    def update_estimator_map(self, keyframe_window, landmarks):
        print("【Backend】: Syncing optimized results back to Estimator...")
        optimized_results = self.active_values

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

        # 确保初始化时状态机是干净的
        self.active_values.clear()
        self.active_kf_gtsam_ids.clear()
        self.marg_factor = None  # 刚初始化时显然没有先验残余
        
        # ---------------------------------------------------------
        # 1. 插入初始帧状态 (Pose, Vel, Bias) 并添加强先验
        # ---------------------------------------------------------
        for i, kf in enumerate(initial_keyframes):
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            self.active_kf_gtsam_ids.append(kf_gtsam_id)

            # 从初始化结果中获取位姿、速度和偏置
            T_wb = gtsam.Pose3(kf.get_global_pose())
            velocity = initial_velocities[i*3 : i*3+3] # initial_velocities是一个扁平化的数组，每3个元素是一个速度向量
            bias = initial_bias # 所有帧使用相同的初始偏置

            # 添加初始估计值
            self.active_values.insert(X(kf_gtsam_id), T_wb)
            self.active_values.insert(V(kf_gtsam_id), velocity)
            self.active_values.insert(B(kf_gtsam_id), bias)

            # 为第一帧添加强先验
            if kf_gtsam_id == 0:
                prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*3 + [1e-2]*3))
                prior_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2e-2] * 3))
                prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1]*3 + [1e-2]*3))

                f_pose = gtsam.PriorFactorPose3(X(0), T_wb, prior_pose_noise)
                f_vel = gtsam.PriorFactorVector(V(0), velocity, prior_vel_noise)
                f_bias = gtsam.PriorFactorConstantBias(B(0), bias, prior_bias_noise)

                graph.add(f_pose)
                graph.add(f_vel)
                graph.add(f_bias)

                # 存进缓存备用
                self.init_priors.add(f_pose)
                self.init_priors.add(f_vel)
                self.init_priors.add(f_bias)
        
        # ---------------------------------------------------------
        # 2. 插入初始路标点状态 (3D Position)
        # ---------------------------------------------------------
        for lm_id, lm_3d_pos in initial_landmarks.items():
            # 致命防御：防 NaN/Inf
            if np.isnan(lm_3d_pos).any() or np.isinf(lm_3d_pos).any():
                print(f"【Backend Warning】: Landmark {lm_id} has NaN/Inf position. Skipped.")
                continue
                
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            self.active_values.insert(L(lm_gtsam_id), lm_3d_pos)

        # ---------------------------------------------------------
        # 3. 压入所有 IMU 预积分因子
        # ---------------------------------------------------------
        for factor_data in initial_imu_factors:
            start_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['start_kf_timestamp'])
            end_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['end_kf_timestamp'])
            
            gtsam_id1 = self._get_kf_gtsam_id(start_kf.get_id())
            gtsam_id2 = self._get_kf_gtsam_id(end_kf.get_id())
            pim = factor_data['imu_preintegration']
            
            graph.add(gtsam.CombinedImuFactor(
                X(gtsam_id1), V(gtsam_id1), X(gtsam_id2), V(gtsam_id2), B(gtsam_id1), B(gtsam_id2), pim))

        # ---------------------------------------------------------
        # 4. 压入所有视觉重投影因子
        # ---------------------------------------------------------
        for kf in initial_keyframes:
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            T_wb = gtsam.Pose3(kf.get_global_pose()) # 提取该帧位姿用于深度计算
            
            for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                if lm_id in initial_landmarks:
                    lm_gtsam_id = self._get_lm_gtsam_id(lm_id)

                    # 确保这个点在 active_values 中存在（可能因 NaN 被跳过）
                    if not self.active_values.exists(L(lm_gtsam_id)):
                        continue

                    current_lm_pos = initial_landmarks[lm_id]

                    # 计算深度并应用降权
                    depth = self._compute_landmark_depth(current_lm_pos, T_wb)
                    weighted_noise = self._get_adaptive_noise(depth, False)

                    factor = gtsam.GenericProjectionFactorCal3_S2(
                        pt_2d, weighted_noise, X(kf_gtsam_id), L(lm_gtsam_id), 
                        self.K, body_P_sensor=self.body_T_cam
                    )
                    graph.add(factor)

        # ---------------------------------------------------------
        # 5. 执行全局 LM 批量优化
        # ---------------------------------------------------------
        print(f"【Backend】: Initializing graph with {graph.size()} factors and {self.active_values.size()} variables...")        
        
        try:
            start_time = time.time()
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, self.active_values, params)
            self.active_values = optimizer.optimize()
            end_time = time.time()
            print(f"【Backend Timer】: Initial optimization took { (end_time - start_time) * 1000:.3f} ms.")
        except RuntimeError as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!! INITIALIZATION FAILED !!!!!!!!!!!!!!")
            print(f"ERROR: {e}")
            return


        # ---------------------------------------------------------
        # 6. 更新并打印最新状态
        # ---------------------------------------------------------
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        latest_gtsam_id = self.next_gtsam_kf_id - 1

        print(f"【Backend】: Latest gtsam_id: {latest_gtsam_id}")
        print(f"【Backend】: Latest optimized state: pose: {latest_pose.matrix()}, velocity: {latest_vel}, bias: {latest_bias}")

        if latest_bias is not None:
            self.latest_bias = latest_bias
        print("【Backend】: Initial graph optimization complete.")

        # 记录优化状态
        new_factors_error = self._log_optimization_error(graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        # ---------------------------------------------------------
        # 7. 后处理：极端情况的初始边缘化 (防超窗)
        # ---------------------------------------------------------
        # 通常 init_window_size <= lag_window_size，但也可能触发超窗
        # while len(self.active_kf_gtsam_ids) > self.lag_window_size:
        oldest_gtsam_id = self.active_kf_gtsam_ids.pop(0)
        keys_to_marg_list = [X(oldest_gtsam_id), V(oldest_gtsam_id), B(oldest_gtsam_id)]
        keys_to_marg = gtsam.KeyVector()
        for k in keys_to_marg_list: keys_to_marg.append(k)

        try:
            self.marg_factor = gtsam.marginalizeOut(graph, self.active_values, keys_to_marg)
        except Exception as e:
            print(f"【Backend Init Error】: Marginalization failed: {e}")
            
        for k in keys_to_marg_list:
            if self.active_values.exists(k):
                self.active_values.erase(k)

    def window_optimize(self, active_kfs, window_imu_factors, window_landmarks,
                             window_visual_factors, new_kf_initial_guess, is_stationary):
        # 获取最新关键帧对象，使用IMU预测值作为初始估计
        old_keyframe = active_kfs[0]
        new_keyframe = active_kfs[-1]
        kf_gtsam_id = self._get_kf_gtsam_id(new_keyframe.get_id())
        T_wb_guess, vel_guess, bias_guess = new_kf_initial_guess

        # ！！！绝对防火墙：过滤掉历史已边缘化的点，阻止它们重返状态机 ！！！
        window_landmarks = {k: v for k, v in window_landmarks.items() if k not in self.marginalized_landmarks}
        window_visual_factors = [f for f in window_visual_factors if f[1] not in self.marginalized_landmarks]

        # ---------------------------------------------------------
        # 1. 向 Active Values 中插入新状态 (仅仅是新帧和新点)
        # ---------------------------------------------------------
        if not self.active_values.exists(X(kf_gtsam_id)):
            self.active_values.insert(X(kf_gtsam_id), T_wb_guess)
            self.active_values.insert(V(kf_gtsam_id), vel_guess)
            self.active_values.insert(B(kf_gtsam_id), bias_guess)
            self.active_kf_gtsam_ids.append(kf_gtsam_id)

        for lm_id, lm_3d_pos in window_landmarks.items():
            # 安全检查：拒绝 NaN/Inf
            if np.isnan(lm_3d_pos).any() or np.isinf(lm_3d_pos).any():
                print(f"🔥 【Backend】[致命警告]: 路标点 L{lm_id} 的初始值无效 (NaN/Inf)！优化即将因此崩溃！")
                continue
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            if not self.active_values.exists(L(lm_gtsam_id)):
                self.active_values.insert(L(lm_gtsam_id), lm_3d_pos)

        # ---------------------------------------------------------
        # 2. 从零构建当前的局部 FactorGraph
        # ---------------------------------------------------------
        current_graph = gtsam.NonlinearFactorGraph()

        # [A] 加入上一次的边缘化因子（DM-VIO逻辑）
        if self.marg_factor is not None:
            current_graph.push_back(self.marg_factor)

        # [A.1] 如果是最开始的几帧（0帧还在窗口内），必须加上绝对先验防漂移
        if X(0) in [X(id) for id in self.active_kf_gtsam_ids]:
            current_graph.push_back(self.init_priors)

        # [B] 压入所有 IMU 预积分因子
        for imu_data in window_imu_factors:
            id1 = self._get_kf_gtsam_id(imu_data['kf_id1'])
            id2 = self._get_kf_gtsam_id(imu_data['kf_id2'])
            pim = imu_data['pim']
            imu_factor = gtsam.CombinedImuFactor(
                X(id1), V(id1), X(id2), V(id2), B(id1), B(id2), pim)
            current_graph.push_back(imu_factor)

        # [C] 压入视觉因子（带防火墙机制）
        bad_landmarks = set()
        valid_visual_factors = []

        # -- 防火墙预检 --
        for kf_id, lm_id, pt_2d in window_visual_factors:
            k_id = self._get_kf_gtsam_id(kf_id)
            l_id = self._get_lm_gtsam_id(lm_id)

            if not self.active_values.exists(X(k_id)) or not self.active_values.exists(L(l_id)):
                continue

            pose = self.active_values.atPose3(X(k_id))
            point = self.active_values.atPoint3(L(l_id))
            
            # 使用严格的基础噪声进行重投影误差检测
            check_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0) 
            temp_factor = gtsam.GenericProjectionFactorCal3_S2(
                pt_2d, check_noise, X(k_id), L(l_id), self.K, body_P_sensor=self.body_T_cam
            )
            
            temp_values = gtsam.Values()
            temp_values.insert(X(k_id), pose)
            temp_values.insert(L(l_id), point)
            
            try:
                error = temp_factor.error(temp_values)
            except:
                error = float('inf')

            # 超过拒绝阈值，标记为坏点
            # 阈值设定：
            # Error = 0.5 * (u-u')^2 / sigma^2
            # 如果 sigma=1, error=50 意味着像素误差 sqrt(100) = 10 像素
            if error > self.rejection_threshold:
                print(f"🔥 [Firewall] 拦截坏因子! KF{kf_id}-LM{lm_id}, Error: {error:.2f}")
                bad_landmarks.add(lm_id)
            else:
                depth = self._compute_landmark_depth(point, pose) if self.use_depth_weight else None
                valid_visual_factors.append((k_id, l_id, pt_2d, depth))

        # -- 正式压入健康的视觉因子 --
        for k_id, l_id, pt_2d, depth in valid_visual_factors:
            # 原始 lm_id 需要反向查找（如果你存了的话），这里简写为不在黑名单即可
            # 只要这个点在任意一帧中被防火墙拉黑，我们就彻底放弃它的所有观测
            original_lm_id = [k for k, v in self.landmark_id_to_gtsam_id.items() if v == l_id][0]
            if original_lm_id in bad_landmarks:
                continue

            # 使用自适应 Huber 核
            # is_new = False # 根据业务逻辑判断是否刚三角化，简单起见可默认为 False 或从外部传入
            # weighted_noise = self._get_adaptive_noise(depth, is_new)
            factor = gtsam.GenericProjectionFactorCal3_S2(
                pt_2d, self.visual_robust_noise, X(k_id), L(l_id), self.K, body_P_sensor=self.body_T_cam
            )
            current_graph.push_back(factor)

        # 清除状态中那些被彻底拉黑的孤立 Landmark
        # 但必须保留 marg_factor 引用的点，否则优化器找不到对应的 key
        # 打印发现marg_factor会保留这个拉黑的点index
        marg_keys = set()
        if self.marg_factor is not None:
            marg_keys = set(self.marg_factor.keys())

        for bad_lm in bad_landmarks:
            bad_l_id = self._get_lm_gtsam_id(bad_lm)
            if L(bad_l_id) in marg_keys:
                continue
            if self.active_values.exists(L(bad_l_id)):
                self.active_values.erase(L(bad_l_id))


        # [D] 零速度更新 (ZUPT)
        if is_stationary:
            zero_velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.03)
            zero_velocity_prior = gtsam.PriorFactorVector(V(kf_gtsam_id), np.zeros(3), zero_velocity_noise)
            current_graph.add(zero_velocity_prior)
            print("【Backend】: Added Zero-Velocity-Update (ZUPT) factor.")

            # # 添加单位位姿因子
            # no_motion_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            #     np.array([0.01, 0.01, 0.01,  # 旋转轴 (roll, pitch, yaw)
            #               0.03, 0.03, 0.03])) # 平移 (x, y, z)
            
            # new_graph.add(gtsam.BetweenFactorPose3(X(last_kf_gtsam_id), X(kf_gtsam_id),      
            #               gtsam.Pose3(), no_motion_pose_noise))
            # print("【Backend】: Added No-Motion Pose Factor.")

        # ---------------------------------------------------------
        # 2.5 清理孤儿变量：移除 active_values 中不被任何因子引用的变量
        # ---------------------------------------------------------
        graph_keys = set()
        for i in range(current_graph.size()):
            factor = current_graph.at(i)
            if factor is not None:
                for key in factor.keys():
                    graph_keys.add(key)

        orphan_keys = [k for k in self.active_values.keys() if k not in graph_keys]
        if orphan_keys:
            orphan_strs = [gtsam.DefaultKeyFormatter(k) for k in orphan_keys]
            print(f"【Backend GC】: 清理 {len(orphan_keys)} 个孤儿变量: {orphan_strs}")
            for k in orphan_keys:
                self.active_values.erase(k)

        # ---------------------------------------------------------
        # 3. 执行局部 LM 优化
        # ---------------------------------------------------------
        try:
            start_time = time.time()
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(current_graph, self.active_values, params)
            self.active_values = optimizer.optimize()
            print(f"【Backend】: Optimization took {(time.time() - start_time) * 1000:.2f} ms")
        except RuntimeError as e:
            print(f"【Backend】: New keyframe: {new_keyframe.get_id()}")
            print(f"【Backend】: Old keyframe: {old_keyframe.get_id()}")
            print(f"!!!!!!!!!! OPTIMIZATION FAILED !!!!!!!!!!!!!!\nERROR: {e}")
            return

        # ---------------------------------------------------------
        # 4. 更新最新状态及记录状态
        # ---------------------------------------------------------
        # 更新最新bias
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        # print(f"【Backend】: Latest optimized state: pose: {latest_pose.matrix()}, velocity: {latest_vel}, bias: {latest_bias}")
        latest_gtsam_id = self.next_gtsam_kf_id - 1
        if latest_bias is not None:
            self.latest_bias = latest_bias

        if latest_pose is None:
             print("【Backend】Critical: Optimization succeeded but state retrieval failed.")
             return

        # 记录优化误差
        new_factors_error = self._log_optimization_error(current_graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        print("【Backend】: Incremental optimization complete.")

        # ---------------------------------------------------------
        # 4. 滑动窗口边缘化 (调用 C++ 环境)
        # ---------------------------------------------------------
        oldest_gtsam_id = self.active_kf_gtsam_ids.pop(0) # 找到最老的帧
        keys_to_marg_list = [X(oldest_gtsam_id), V(oldest_gtsam_id), B(oldest_gtsam_id)] # 需要边缘化的最老帧状态Key

        # 获取 oldest_gtsam_id 对应的原始 kf_id
        oldest_kf_id = [k for k, v in self.kf_id_to_gtsam_id.items() if v == oldest_gtsam_id][0]

        # 🔥 找出需要【连坐】边缘化的路标点（即第一次观测发生在最老帧的点）
        lm_first_obs = {}
        for kf_id, lm_id, pt_2d in window_visual_factors:
            if lm_id not in lm_first_obs:
                lm_first_obs[lm_id] = kf_id
            else:
                lm_first_obs[lm_id] = min(lm_first_obs[lm_id], kf_id)

        # 只有生命周期起源于最老帧的点，才会被选中
        lms_to_marg_ids = [lm for lm, first_kf in lm_first_obs.items() if first_kf == oldest_kf_id]

        # 制定死亡名单
        keys_to_marg_list = [X(oldest_gtsam_id), V(oldest_gtsam_id), B(oldest_gtsam_id)]
        for lm_id in lms_to_marg_ids:
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            if self.active_values.exists(L(lm_gtsam_id)):
                keys_to_marg_list.append(L(lm_gtsam_id))
        
        # 2. 构建边缘化图
        marg_graph = gtsam.NonlinearFactorGraph()

        # 1. 继承上一次的先验
        if self.marg_factor is not None:
            marg_graph.push_back(self.marg_factor)

        # 2. 如果第0帧刚刚被砍，把初始化先验加上，传承火种
        if oldest_gtsam_id == 0:
            marg_graph.push_back(self.init_priors)

        # 3. 找出【仅与最老帧相连】的 IMU 因子并加入
        for imu_data in window_imu_factors:
            id1 = self._get_kf_gtsam_id(imu_data['kf_id1'])
            if id1 == oldest_gtsam_id:
                id2 = self._get_kf_gtsam_id(imu_data['kf_id2'])
                pim = imu_data['pim']
                imu_factor = gtsam.CombinedImuFactor(
                    X(id1), V(id1), X(id2), V(id2), B(id1), B(id2), pim)
                marg_graph.push_back(imu_factor)

        # 4. 是否需要添加观测数量少于一定值的视觉因子？（感觉不用）

        # 🔥 核心修正：把待边缘化路标点的【所有】视觉观测加进来，提取先验防漂移！
        for kf_id, lm_id, pt_2d in window_visual_factors:
            if lm_id in lms_to_marg_ids:
                k_id = self._get_kf_gtsam_id(kf_id)
                l_id = self._get_lm_gtsam_id(lm_id)
                if self.active_values.exists(X(k_id)) and self.active_values.exists(L(l_id)):
                    pose = self.active_values.atPose3(X(k_id))
                    point = self.active_values.atPoint3(L(l_id))
                    
                    # 建议在边缘化时使用固定基础噪声，防止 Huber 破坏舒尔补的线性化近似
                    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5) 
                    factor = gtsam.GenericProjectionFactorCal3_S2(
                        pt_2d, noise, X(k_id), L(l_id), self.K, body_P_sensor=self.body_T_cam)
                    marg_graph.push_back(factor)

        # 注意：我们【故意不加】任何视觉因子！让最老帧的视觉观测随风飘散
        # 这样生成的 marg_factor 将绝对稀疏（只包含相邻的 Pose/Vel/Bias）

        # 将 Python list 转换为 GTSAM/C++ 接受的 KeyVector
        keys_to_marg = gtsam.KeyVector()
        for k in keys_to_marg_list:
            keys_to_marg.append(k)

        # 直接调用 C++ 的 marginalizeOut
        # 注意：这里的 marg_graph 包含了本次优化的所有IMU约束和上一次的先验，
        # active_values 已经是刚才优化后的最新结果了，FEJ 极度精确！
        try:
            new_marg_factor = gtsam.marginalizeOut(marg_graph, self.active_values, keys_to_marg)
            self.marg_factor = new_marg_factor # 更新先验
            print(f"【Backend】: Marginalized old frame X({oldest_gtsam_id}) successfully.")
        except Exception as e:
            print(f"【Backend】: Marginalization Error: {e}")

        # 从 active_values 中彻底删除已被边缘化的变量
        for k in keys_to_marg_list:
            if self.active_values.exists(k):
                self.active_values.erase(k)

        # 🔥 绝对封杀：告诉后端，这些点已经化为先验，永远不要再出现！
        for lm_id in lms_to_marg_ids:
            self.marginalized_landmarks.add(lm_id)



    def _log_optimization_error(self, current_full_graph):
        try:
            optimized_result = self.active_values
            new_factors_error = current_full_graph.error(optimized_result)

            print(f"【Backend】优化误差统计: 本轮全局误差 = {new_factors_error:.4f}")

            # ======================= DETAILED FACTOR ERROR LOGGING =======================
            debug_start_frame = 0 # 设为0以立即开始打印
            latest_gtsam_id = self.next_gtsam_kf_id - 1
            if latest_gtsam_id >= debug_start_frame:
                print("\n" + "="*40 + f" DETAILED ERROR ANALYSIS (Frame {latest_gtsam_id}) " + "="*40)
                
                # 遍历图中的所有因子
                for i in range(current_full_graph.size()):
                    factor = current_full_graph.at(i)
                    if factor is None: 
                        continue
                        
                    try:
                        error = factor.error(optimized_result)
                        if error > 10.0: 
                            factor_type = factor.__class__.__name__
                            print(f"  - Factor {i}: Error = {error:.4f}, Type = {factor_type}")
                            keys = factor.keys()
                            key_str = ", ".join([gtsam.DefaultKeyFormatter(key) for key in keys])
                            print(f"    Keys: [{key_str}]")
                            # pass # 生产环境中可以关掉打印，或者写入日志
                            
                    except Exception as e_factor:
                        print(f"  - Factor {i}: 无法计算误差或获取Keys. Error: {e_factor}")

                print("="*100 + "\n")
            # ===========================================================================
            
            return new_factors_error
            
        except Exception as e:
            print(f"[Error][Backend] 计算优化误差时出错: {e}")
            return -1.0
        
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
