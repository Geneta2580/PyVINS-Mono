import queue
import numpy as np
from utils.geometry import world_to_inv_depth, inv_depth_to_world
import gtsam
import gtsam_unstable
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
        self.lag_window_size = config.get('lag_window_size', 9) # 优化器的滑窗
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01) 
        parameters.relinearizeSkip = 1
        self.smoother = IncrementalFixedLagSmoother(self.lag_window_size, parameters) # 自动边缘化
        
        # 后端黑名单
        self.blacklisted_landmarks = set()

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

        # 状态与id管理
        self.kf_id_to_gtsam_id = {}
        self.landmark_id_to_gtsam_id = {}
        self.next_gtsam_kf_id = 0
        self.factor_indices_to_remove = []

        # 记录landmark锚点帧ID
        self.landmark_anchor_kf_id = {}

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

    def _print_new_factors(self, graph, graph_name="New Factors"):
        """
        打印图中所有因子的详细信息
        """
        print(f"\n{'='*80}")
        print(f"【{graph_name}】: Total {graph.size()} factors")
        print(f"{'='*80}")
        
        factor_counts = {}
        for i in range(graph.size()):
            factor = graph.at(i)
            if factor is None:
                continue
                
            factor_type = factor.__class__.__name__
            
            # 统计因子类型
            if factor_type not in factor_counts:
                factor_counts[factor_type] = 0
            factor_counts[factor_type] += 1
            
            # 获取因子连接的变量
            keys = factor.keys()
            key_strs = [gtsam.DefaultKeyFormatter(key) for key in keys]
            
            # 打印每个因子的详细信息
            print(f"  Factor {i:3d}: {factor_type:40s} -> [{', '.join(key_strs)}]")
        
        # 打印统计信息
        print(f"\n【{graph_name} Summary】:")
        for factor_type, count in sorted(factor_counts.items()):
            print(f"  {factor_type:40s}: {count:4d}")
        print(f"{'='*80}\n")

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
            anchor_kf_id = self.landmark_anchor_kf_id.get(lm_id) # 获取该点的锚点帧ID
            if gtsam_id is not None and optimized_results.exists(L(gtsam_id)):
                # 1. 从优化结果中获取最新的3D坐标
                optimized_position = optimized_results.atPoint3(L(gtsam_id))
                
                if np.linalg.norm(optimized_position) > 1e4: # 极远点
                    print(f"【Backend】: 极远点: {lm_id}")
                    self.remove_stale_landmarks([lm_id], [lm_id], [], None)
                    continue

                if optimized_position[2] < 1e-4:
                    print(f"【Backend】: 深度为负的路标点: {lm_id}")
                    self.remove_stale_landmarks([lm_id], [lm_id], [], None)
                    continue
                
                # 逆深度坐标系到世界坐标系
                # 获取锚点帧的gtsam_id
                anchor_gtsam_id = self.kf_id_to_gtsam_id.get(anchor_kf_id)
                if anchor_gtsam_id is None or not optimized_results.exists(X(anchor_gtsam_id)):
                    continue

                anchor_T_wb = optimized_results.atPose3(X(anchor_gtsam_id))
                anchor_T_wc_gtsam = anchor_T_wb.compose(self.body_T_cam)
                anchor_T_wc_np = np.asarray(anchor_T_wc_gtsam.matrix())
                world_point = inv_depth_to_world(optimized_position, anchor_T_wc_np)

                # 2. 调用对象的方法来更新其内部状态
                landmark_obj.set_triangulated(world_point)
                # print(f"【Backend】: Updated landmark {lm_id} to {optimized_position}")

    def remove_stale_landmarks(self, unhealty_lm_ids, unhealty_lm_ids_depth, 
                                unhealty_lm_ids_reproj, oldest_kf_id_in_window):
        print(f"【Backend】: 接收到移除 {len(unhealty_lm_ids)} 个陈旧路标点的指令。")
        if not unhealty_lm_ids:
            return
        
        # 只删除ID映射，阻止这些landmark再次被添加到图中
        for lm_id in unhealty_lm_ids:
            if lm_id not in self.blacklisted_landmarks:
                self.blacklisted_landmarks.add(lm_id)
                print(f"【Backend】: 已将 landmark {lm_id} 加入黑名单")

            if lm_id in self.landmark_id_to_gtsam_id:
                del self.landmark_id_to_gtsam_id[lm_id]
                print(f"【Backend】: 已移除 landmark {lm_id} 的ID映射")

            if lm_id in self.landmark_anchor_kf_id:
                del self.landmark_anchor_kf_id[lm_id]
                print(f"【Backend】: 已移除 landmark {lm_id} 的锚点帧ID")

        print(f"【Backend】: 成功标记 {len(unhealty_lm_ids)} 个路标点为待清理状态")
        print(f"【Backend】: Fixed-Lag Smoother 将在滑窗移动时自动清理这些landmark")

        # # 删除因子逻辑
        # graph = self.smoother.getFactors()
        # factor_indices_to_remove = []
        # unhealty_lm_keys = {L(self._get_lm_gtsam_id(lm_id)) for lm_id in unhealty_lm_ids}
        # unhealty_lm_keys_depth = {L(self._get_lm_gtsam_id(lm_id)) for lm_id in unhealty_lm_ids_depth}
        # unhealty_lm_keys_reproj = {L(self._get_lm_gtsam_id(lm_id)) for lm_id in unhealty_lm_ids_reproj}

        # oldest_gtsam_key = None
        # if oldest_kf_id_in_window is not None and oldest_kf_id_in_window in self.kf_id_to_gtsam_id:
        #     oldest_gtsam_key = X(self._get_kf_gtsam_id(oldest_kf_id_in_window))
        #     print(f"【Backend】: 最旧的关键帧的gtsam_id: {oldest_gtsam_key}")

        # # 收集需要删除的因子
        # for i in range(graph.size()):
        #     factor = graph.at(i)
        #     # print(f"【Backend】: 因子类型: {factor.__class__.__name__}")
        #     if factor is None: continue
            
        #     # 只删除投影因子，绝不删除边缘化因子、IMU因子等
        #     target_types = (gtsam_unstable.InvDepthFactorVariant3a, gtsam_unstable.InvDepthFactorVariant3b)
        #     if not isinstance(factor, target_types):
        #         continue
            
        #     f_keys = factor.keys()
        #     for key in f_keys:
        #         if key in unhealty_lm_keys_depth or key in unhealty_lm_keys_reproj:
        #             key_str = ", ".join([gtsam.DefaultKeyFormatter(k) for k in f_keys])
        #             print(f"  [标记删除] Index: {i}, 连接: [{key_str}]")
        #             factor_indices_to_remove.append(i)
        #             break

        # # 关键修改：只删除因子，不要尝试操作变量的时间戳
        # if factor_indices_to_remove:
        #     empty_graph = gtsam.NonlinearFactorGraph()
        #     empty_values = gtsam.Values()
        #     # empty_stamps = FixedLagSmootherKeyTimestampMap()
        #     empty_stamps = {}
            
        #     self.smoother.update(empty_graph, empty_values, empty_stamps, factor_indices_to_remove)
        #     print(f"【Backend】: 成功移除 {len(factor_indices_to_remove)} 个深度为负或重投影误差过大的路标点的因子")

        # # 删除ID映射 - 修正：只删除那些实际删除了因子的landmark
        # for lm_id in unhealty_lm_ids:  # 改为 unhealty_lm_ids_depth
        #     if lm_id in self.landmark_id_to_gtsam_id:
        #         del self.landmark_id_to_gtsam_id[lm_id]

        # print(f"【Backend】: 成功移除 {len(unhealty_lm_ids)} 个路标点的因子")
        

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

        # 添加所有初始IMU因子
        for factor_data in initial_imu_factors:
            start_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['start_kf_timestamp'])
            end_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['end_kf_timestamp'])
            gtsam_id1 = self._get_kf_gtsam_id(start_kf.get_id())
            gtsam_id2 = self._get_kf_gtsam_id(end_kf.get_id())
            pim = factor_data['imu_preintegration']
            graph.add(gtsam.CombinedImuFactor(X(gtsam_id1), V(gtsam_id1), X(gtsam_id2), V(gtsam_id2), B(gtsam_id1), B(gtsam_id2), pim))


        # 添加所有初始路标点变量和视觉因子
        last_gtsam_id = self._get_kf_gtsam_id(initial_keyframes[-1].get_id())
        for kf in initial_keyframes:
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            T_wb = gtsam.Pose3(kf.get_global_pose())
            T_wc = T_wb.compose(self.body_T_cam)
            T_wc_np = np.asarray(T_wc.matrix())

            for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                if lm_id not in initial_landmarks: continue

                lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
                lm_3d_pos = initial_landmarks[lm_id]

                # 简单噪声
                weighted_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)

                # 路标点未进入图
                if not estimates.exists(L(lm_gtsam_id)):
                    # 初始化路标点变量
                    inv_params = world_to_inv_depth(lm_3d_pos, T_wc_np)

                    if inv_params is None or np.isnan(inv_params).any():
                        continue

                    # 设置当前帧为锚点帧
                    self.landmark_anchor_kf_id[lm_id] = kf.get_id() # TODO： 观察这里的ID

                    estimates.insert(L(lm_gtsam_id), inv_params)
                    initial_window_stamps[L(lm_gtsam_id)] = float(last_gtsam_id)

                    # 添加逆深度锚点因子
                    inv_depth_factor3a = gtsam_unstable.InvDepthFactorVariant3a(
                        X(kf_gtsam_id), L(lm_gtsam_id), pt_2d, 
                        self.K, weighted_noise, self.body_T_cam
                    )
                    graph.add(inv_depth_factor3a)

                # 路标点已经在图中
                else:
                    anchor_kf_id = self.landmark_anchor_kf_id[lm_id]
                    
                    if anchor_kf_id is None:
                        continue

                    anchor_gtsam_id = self._get_kf_gtsam_id(anchor_kf_id)

                    inv_depth_factor3b = gtsam_unstable.InvDepthFactorVariant3b(
                        X(anchor_gtsam_id), X(kf_gtsam_id), L(lm_gtsam_id), pt_2d, 
                        self.K, weighted_noise, self.body_T_cam
                    )
                    graph.add(inv_depth_factor3b)

        # 执行iSAM2的第一次更新（批量模式）
        print(f"【Backend】: Initializing iSAM2 with {graph.size()} new factors and {estimates.size()} new values...")
        
        # 打印所有新因子
        self._print_new_factors(graph, "Initialization Factors")
        
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
        # 检查上一帧是否边缘化
        if not current_isam_values.exists(X(last_kf_gtsam_id)) and not new_estimates.exists(X(last_kf_gtsam_id)):
            print(f"【CRITICAL ERROR】试图连接已经边缘化的上一帧 {last_kf_gtsam_id}！重置系统或扩展滑窗！")
            # 这里通常应该触发系统复位，因为IMU约束链断了
            return

        pim = new_imu_factors['imu_preintegration']
        imu_factor = gtsam.CombinedImuFactor(
            X(last_kf_gtsam_id), V(last_kf_gtsam_id), X(kf_gtsam_id), V(kf_gtsam_id),
            B(last_kf_gtsam_id), B(kf_gtsam_id), pim)
        new_graph.add(imu_factor)

        # ======================= 视觉因子处理 (带缓存机制) =======================
        # 【缓存字典】: 用于暂存新路标点及其因子
        # 结构: { lm_id: { 'value': Point3, 'factors': [factor1, factor2, ...], 'anchor_kf': kf_id } }
        new_landmark_buffer = {}

        # [Helper] 获取任意关键帧Pose的辅助函数
        def get_pose_for_kf(target_kf_id):
            target_gtsam_id = self._get_kf_gtsam_id(target_kf_id)
            # 1. 如果是当前正在优化的新帧 -> 从 guess 取
            if target_kf_id == new_keyframe.get_id():
                return T_wb_guess
            # 2. 如果是历史帧 -> 从 ISAM 结果取
            elif current_isam_values.exists(X(target_gtsam_id)):
                return current_isam_values.atPose3(X(target_gtsam_id))
            # 3. 如果是刚加入new_estimates但不是当前帧(极少见) -> 从 new_estimates 取
            elif new_estimates.exists(X(target_gtsam_id)):
                return new_estimates.atPose3(X(target_gtsam_id))
            else:
                return None # 帧已丢失或被边缘化

        for kf_id, lm_id, pt_2d in new_visual_factors:
            # 关键检查：如果landmark的ID映射已被删除，说明它已被标记为待清理，跳过
            if lm_id not in self.landmark_id_to_gtsam_id:
                continue

            # 检查后端黑名单
            if lm_id in self.blacklisted_landmarks:
                continue
            
            # 获取当前关键帧和路标点的映射ID（新点会直接添加到映射）
            curr_kf_gtsam_id = self._get_kf_gtsam_id(kf_id)
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)

            # 判断点状态
            is_lm_in_graph = current_isam_values.exists(L(lm_gtsam_id))
            is_lm_in_buffer = lm_id in new_landmark_buffer
            # is_lm_in_new = new_estimates.exists(L(lm_gtsam_id))

            # ---------------- Case A: 这是个全新的点 (创建 3a 因子) ----------------
            if not is_lm_in_graph and not is_lm_in_buffer:
                if lm_id not in new_landmarks: continue
                lm_3d_pos_w = new_landmarks[lm_id]

                # 获取该观测帧(Anchor)的真实位姿
                kf_pose_body = get_pose_for_kf(kf_id) 
                if kf_pose_body is None: continue 

                # 计算 Anchor 帧的相机位姿
                T_wc_anchor = kf_pose_body.compose(self.body_T_cam)
                T_wc_anchor_np = np.asarray(T_wc_anchor.matrix())

                # 逆深度必须相对于 Anchor 帧计算
                inv_params = world_to_inv_depth(lm_3d_pos_w, T_wc_anchor_np)
                if inv_params is None or np.isnan(inv_params).any(): continue

                # 添加逆深度锚点3a因子
                inv_depth_factor3a = gtsam_unstable.InvDepthFactorVariant3a(
                    X(curr_kf_gtsam_id), L(lm_gtsam_id), 
                    pt_2d, self.K, self.visual_robust_noise, self.body_T_cam
                )

                # 简单的 Chi2 检查
                kf_pose = get_pose_for_kf(kf_id) # 获取该观测帧的真实位姿
                if kf_pose is None: continue # 如果观测帧已经边缘化，无法建立约束

                temp_val = gtsam.Values()
                temp_val.insert(X(curr_kf_gtsam_id), kf_pose)
                temp_val.insert(L(lm_gtsam_id), inv_params)

                if inv_depth_factor3a.error(temp_val) < 50.0:
                    new_landmark_buffer[lm_id] = {
                        'value': inv_params,
                        'factors': [inv_depth_factor3a],
                        'anchor_kf': kf_id,
                        "gtsam_id": lm_gtsam_id
                    }
                    # 记录锚点关键帧ID
                    self.landmark_anchor_kf_id[lm_id] = kf_id

            # ---------------- Case B: 这是个在 Buffer 中的新点 (创建 3b 因子) ----------------
            elif is_lm_in_buffer:
                # 获取锚点信息
                buffer_data = new_landmark_buffer[lm_id]
                anchor_kf_id = buffer_data['anchor_kf']
                anchor_gtsam_id = self._get_kf_gtsam_id(anchor_kf_id)

                # 创建3b因子
                inv_depth_factor3b = gtsam_unstable.InvDepthFactorVariant3b(
                    X(anchor_gtsam_id), X(curr_kf_gtsam_id), L(lm_gtsam_id), pt_2d, 
                    self.K, self.visual_robust_noise, self.body_T_cam
                )
                
                # Chi2 检查
                kf_pose = get_pose_for_kf(kf_id) # 获取观测帧位姿
                anchor_pose = get_pose_for_kf(anchor_kf_id) # 获取锚点帧位姿
                
                if kf_pose is not None and anchor_pose is not None:
                    try:
                        temp_val = gtsam.Values()
                        temp_val.insert(X(curr_kf_gtsam_id), kf_pose)
                        temp_val.insert(L(lm_gtsam_id), buffer_data['value'])
                        temp_val.insert(X(anchor_gtsam_id), anchor_pose)
                        
                        if inv_depth_factor3b.error(temp_val) < 150.0: 
                            buffer_data['factors'].append(inv_depth_factor3b)

                    except Exception:
                        pass
            
            # ---------------- Case C: 这是个已经在优化图中的老点 (3b 因子) ----------------
            elif is_lm_in_graph:
                # 检查观测帧存活（防御策略）
                if not current_isam_values.exists(X(curr_kf_gtsam_id)) and not new_estimates.exists(X(curr_kf_gtsam_id)):
                    continue

                # 获取锚点信息
                anchor_kf_id = self.landmark_anchor_kf_id[lm_id]
                if anchor_kf_id is None: continue
                anchor_gtsam_id = self._get_kf_gtsam_id(anchor_kf_id)

                # 检查锚点存活（如果锚点帧已经边缘化，则跳过）
                if not current_isam_values.exists(X(anchor_gtsam_id)) and not new_estimates.exists(X(anchor_gtsam_id)):
                    continue
                
                # 创建3b因子
                inv_depth_factor3b = gtsam_unstable.InvDepthFactorVariant3b(
                    X(anchor_gtsam_id), X(curr_kf_gtsam_id), L(lm_gtsam_id), pt_2d, 
                    self.K, self.visual_robust_noise, self.body_T_cam)

                # 老点直接加图
                new_graph.add(inv_depth_factor3b)
                new_window_stamps[L(lm_gtsam_id)] = float(curr_kf_gtsam_id)

        # ======================= 提交新点因子 =======================
        # 遍历 Buffer，只有约束足够的点才真正加入系统
        valid_new_count = 0
        for lm_id, data in new_landmark_buffer.items():
            factors = data['factors']

            # 只有当因子数量 >= 2 时才提交
            if len(factors) >= 2:
                lm_gtsam_id = data['gtsam_id']

                # 插入变量顶点并更新滑窗时间戳
                new_estimates.insert(L(lm_gtsam_id), data['value'])
                new_window_stamps[L(lm_gtsam_id)] = float(kf_gtsam_id)

                # 插入所有缓存因子
                for f in factors:
                    new_graph.add(f)

                valid_new_count += 1
            else:
                if lm_id in self.landmark_anchor_kf_id:
                    del self.landmark_anchor_kf_id[lm_id]
                print(f"【Backend】Drop candidate lm {lm_id}: Only {len(factors)} factors (Unconstrained).")

        # print(f"【Backend】Commited {valid_new_count} / {len(new_landmark_buffer)} new landmarks.")

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
         
        # 打印所有新因子
        self._print_new_factors(new_graph, f"Incremental Factors (KF {new_keyframe.get_id()})")
        
        try:
            start_time = time.time()
            self.smoother.update(new_graph, new_estimates, new_window_stamps)
            end_time = time.time()
            print(f"【Backend Timer】: Incremental optimization took { (end_time - start_time) * 1000:.3f} ms.")

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

        # 记录优化误差
        new_factors_error = self._log_optimization_error(new_graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        print("【Backend】: Incremental optimization complete.")


    def _log_optimization_error(self, new_factors_graph):
        try:
            # 1. 获取当前滑窗内的有效变量值
            optimized_result = self.smoother.calculateEstimate()
            
            # 2. 安全计算本轮新增因子的误差
            # 只有当新因子涉及的所有变量都在 active values 里时才计算
            # (通常 optimize_incremental 里的逻辑已经保证了这一点，但加一层 try-catch 更稳妥)
            try:
                new_factors_error = new_factors_graph.error(optimized_result)
            except RuntimeError:
                # 如果新因子连接了已经被边缘化的变量（理论上不应发生），设为 -1
                new_factors_error = -1.0

            current_full_graph = self.smoother.getFactors()

            print(f"【Backend】优化误差统计: "
                  f"本轮新增因子误差 = {new_factors_error:.4f}")

            # ======================= DETAILED FACTOR ERROR LOGGING =======================
            debug_start_frame = 0 
            latest_gtsam_id = self.next_gtsam_kf_id - 1
            
            if latest_gtsam_id >= debug_start_frame:
                print("\n" + "="*40 + f" DETAILED ERROR ANALYSIS (Frame {latest_gtsam_id}) " + "="*40)
                
                # 遍历图中的所有因子
                for i in range(current_full_graph.size()):
                    factor = current_full_graph.at(i)
                    if factor is None: 
                        continue
                    
                    try:
                        # ==================== [修复开始] ====================
                        # 核心修复：检查因子引用的所有 Key 是否都存在于当前 Values 中
                        # 如果涉及了被边缘化的旧变量 (例如 x0)，则跳过计算，防止崩溃
                        keys = factor.keys()
                        all_keys_exist = True
                        for key in keys:
                            if not optimized_result.exists(key):
                                all_keys_exist = False
                                break
                        
                        if not all_keys_exist:
                            # 这是一个连接到已被边缘化变量的因子（通常是Prior或旧观测），跳过
                            continue
                        # ==================== [修复结束] ====================

                        # 计算这个特定因子的误差
                        error = factor.error(optimized_result)
                        
                        # 打印误差大于阈值的因子
                        if error > 10.0: 
                            factor_type = factor.__class__.__name__
                            # print(f"  - Factor {i}: Error = {error:.4f}, Type = {factor_type}")
                            
                            key_str = ", ".join([gtsam.DefaultKeyFormatter(key) for key in keys])
                            print(f"  - Factor {i} [{factor_type}]: Error={error:.2f}, Keys=[{key_str}]")
                           
                    except Exception as e_factor:
                        # 依然保留这个捕获以防万一
                        pass # 忽略打印错误，保持主线程运行

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
