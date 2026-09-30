import time
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B, L

from utils.debug import Debugger
from datatype.landmark import LandmarkStatus

# 导入你的手搓数学引擎 (确保在 utils/geometry.py 中已经实现了该函数及雅可比解析函数)
from utils.geometry import build_structureless_hessian

class Backend:
    def __init__(self, global_central_map, config, imu_processor):
        self.global_central_map = global_central_map
        self.config = config

        # 滑窗数据结构
        self.active_values = gtsam.Values()
        self.active_kf_gtsam_ids = []
        self.marg_factor = None

        # 初始化先验因子缓存
        self.init_priors = gtsam.NonlinearFactorGraph()

        # 噪声因子
        self.visual_noise_sigma = config.get('visual_noise_sigma', 2.0)
        # 注意：在使用手搓引擎后，鲁棒核将在 Numpy 中通过权重动态实现，
        # GTSAM 中的视觉噪声模型将不再直接参与视觉因子的优化，但仍保留以防他用。

        self.rejection_threshold = config.get('rejection_threshold', 400.0)

        # 状态与id管理
        self.kf_id_to_gtsam_id = {}
        self.landmark_id_to_gtsam_id = {}
        self.next_gtsam_kf_id = 0

        # 获取相机内、外参
        cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)
        self.K_mat = cam_intrinsics
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
        self.logger = Debugger(self.config, file_prefix="backend_state", column_names=log_columns)

        # 用于记录历史静止帧，防止 ZUPT 断层跳变
        self.stationary_kf_gtsam_ids = set()

        # 🔥 手搓无结构BA的核心：后端在外部自己维护 3D 点状态
        # 格式: { lm_gtsam_id: np.array([x, y, z]) }
        self.landmark_states = {}

    def _get_kf_gtsam_id(self, kf_id):
        if kf_id not in self.kf_id_to_gtsam_id:
            self.kf_id_to_gtsam_id[kf_id] = self.next_gtsam_kf_id
            self.next_gtsam_kf_id += 1
        return self.kf_id_to_gtsam_id[kf_id]

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

        for kf in keyframe_window:
            gtsam_id = self.kf_id_to_gtsam_id.get(kf.get_id())
            if gtsam_id is not None and optimized_results.exists(X(gtsam_id)):
                pose_w_b = optimized_results.atPose3(X(gtsam_id))
                kf.set_global_pose(pose_w_b.matrix())

        for lm_id, landmark_obj in landmarks.items():
            gtsam_id = self._get_lm_gtsam_id(lm_id)
            if gtsam_id in self.landmark_states:
                optimized_position = self.landmark_states[gtsam_id]
                landmark_obj.set_triangulated(optimized_position)

    def remove_stale_landmarks(self, unhealty_lm_ids, unhealty_lm_ids_depth, 
                                unhealty_lm_ids_reproj, oldest_kf_id_in_window):
        print(f"【Backend】: 接收到移除 {len(unhealty_lm_ids)} 个陈旧路标点的指令。")
        if not unhealty_lm_ids:
            return

        for lm_id in unhealty_lm_ids:
            if lm_id in self.landmark_id_to_gtsam_id:
                l_id = self.landmark_id_to_gtsam_id[lm_id]
                if l_id in self.landmark_states:
                    del self.landmark_states[l_id]
                del self.landmark_id_to_gtsam_id[lm_id]

    def _compute_landmark_depth(self, lm_3d_pos, kf_pose):
        """物理级深度检测：相机系下的Z轴坐标"""
        lm_pos_w = np.array(lm_3d_pos)
        T_w_b = kf_pose.matrix()
        P_b = np.linalg.inv(T_w_b) @ np.append(lm_pos_w, 1.0)
        T_c_b = self.cam_T_body.matrix()
        P_c = T_c_b @ P_b
        return P_c[2]

    # =========================================================================
    # [数学引擎接口] Iterative BA
    # =========================================================================

    def _create_custom_hessian_factor(self, symbols, values, H, v, f_err):
        """将纯 Pose 的增量信息包装为 GTSAM 可接受的 Factor，并补齐常数项误差"""
        info_expand = np.zeros((H.shape[0] + 1, H.shape[1] + 1))
        info_expand[:-1, :-1] = H
        info_expand[:-1, -1] = v.flatten()
        info_expand[-1, :-1] = v.flatten()
        info_expand[-1, -1] = f_err # 🔥 真实的常数平方误差

        dims = [6] * len(symbols)
        h_f = gtsam.HessianFactor(symbols, dims, info_expand)
        l_c = gtsam.LinearContainerFactor(h_f, values)
        return l_c

    def _run_iterative_ba(self, base_graph, valid_visual_factors, kf_ids, lm_ids, max_iters=2):
        """外循环迭代：消除线性化点漂移，手动 Retract 3D点"""
        current_opt_values = gtsam.Values(self.active_values)
        initial_visual_error = 0.0 

        for iteration in range(max_iters):
            # 1. 提取当前状态字典，喂给 Numpy 引擎
            kf_states_dict = {kf: current_opt_values.atPose3(X(kf)) for kf in kf_ids}
            lm_states_dict = {lm: self.landmark_states[lm] for lm in lm_ids if lm in self.landmark_states}
            
            # 2. 调用手搓的数学引擎！
            H_marg, b_marg, H_ll_inv, H_pl, b_l, f_err, valid_lm_ids = build_structureless_hessian(
                kf_states_dict, lm_states_dict, valid_visual_factors, self.K, self.body_T_cam, self.visual_noise_sigma)
            
            # 记录首次迭代的真实误差
            if iteration == 0:
                initial_visual_error = f_err
                
            iter_graph = gtsam.NonlinearFactorGraph(base_graph)
            
            symbols = gtsam.KeyVector()
            for kf_id in kf_ids:
                symbols.append(X(kf_id))
                
            # 3. 将降维后的信息矩阵塞给 GTSAM 求解
            if len(valid_lm_ids) > 0:
                vis_factor = self._create_custom_hessian_factor(symbols, current_opt_values, H_marg, b_marg, f_err)
                iter_graph.push_back(vis_factor)
            
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(iter_graph, current_opt_values, params)
            new_values = optimizer.optimize()
            
            # 4. 🔥 绝妙的流形反传 (Retract)
            Delta_X = np.zeros((6 * len(kf_ids), 1))
            for idx, kf_id in enumerate(kf_ids):
                pose_old = current_opt_values.atPose3(X(kf_id))
                pose_new = new_values.atPose3(X(kf_id))
                # 计算切空间增量
                xi = gtsam.Pose3.Logmap(pose_old.inverse() * pose_new)
                Delta_X[idx*6 : (idx+1)*6, 0] = xi

            # 基于位姿的扰动，一步反解出路标点的最优增量
            Delta_L = H_ll_inv @ (b_l - H_pl.T @ Delta_X)
            
            # 安全更新点坐标
            for idx, l_id in enumerate(valid_lm_ids):
                if l_id in self.landmark_states:
                    dl = Delta_L[idx*3 : (idx+1)*3, 0]
                    self.landmark_states[l_id] += dl # numpy 直接向量相加！
                    
            current_opt_values = new_values

        return current_opt_values, initial_visual_error

    # =========================================================================
    # [标准生命周期]
    # =========================================================================

    def initialize_optimize(self, initial_keyframes, initial_imu_factors, initial_landmarks, initial_velocities, initial_bias):
        print("【Backend】: Initializing optimize...")
        base_graph = gtsam.NonlinearFactorGraph()

        self.active_values.clear()
        self.active_kf_gtsam_ids.clear()
        self.marg_factor = None  
        self.stationary_kf_gtsam_ids.clear()
        self.landmark_states.clear()
        
        # 1. 插入初始帧状态 
        for i, kf in enumerate(initial_keyframes):
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            self.active_kf_gtsam_ids.append(kf_gtsam_id)

            T_wb = gtsam.Pose3(kf.get_global_pose())
            velocity = initial_velocities[i*3 : i*3+3]
            bias = initial_bias

            self.active_values.insert(X(kf_gtsam_id), T_wb)
            self.active_values.insert(V(kf_gtsam_id), velocity)
            self.active_values.insert(B(kf_gtsam_id), bias)

            if kf_gtsam_id == 0:
                base_graph.add(gtsam.PriorFactorPose3(X(0), T_wb, gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*3 + [1e-2]*3))))
                base_graph.add(gtsam.PriorFactorVector(V(0), velocity, gtsam.noiseModel.Diagonal.Sigmas(np.array([2e-2] * 3))))
                base_graph.add(gtsam.PriorFactorConstantBias(B(0), bias, gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1]*3 + [1e-2]*3))))

        # 2. 插入初始路标点状态 (Numpy数组存储)
        for lm_id, lm_3d_pos in initial_landmarks.items():
            if np.isnan(lm_3d_pos).any() or np.isinf(lm_3d_pos).any():
                continue
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            self.landmark_states[lm_gtsam_id] = np.array(lm_3d_pos)

        # 3. 压入所有 IMU 预积分因子
        for factor_data in initial_imu_factors:
            start_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['start_kf_timestamp'])
            end_kf = next(kf for kf in initial_keyframes if kf.get_timestamp() == factor_data['end_kf_timestamp'])
            gtsam_id1, gtsam_id2 = self._get_kf_gtsam_id(start_kf.get_id()), self._get_kf_gtsam_id(end_kf.get_id())
            
            base_graph.add(gtsam.CombinedImuFactor(
                X(gtsam_id1), V(gtsam_id1), X(gtsam_id2), V(gtsam_id2), B(gtsam_id1), B(gtsam_id2), factor_data['imu_preintegration']))

        # 4. 收集观测数据，喂给自定义引挚
        valid_visual_factors = []
        active_lm_ids = set()
        for kf in initial_keyframes:
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                if lm_id in initial_landmarks:
                    l_id = self._get_lm_gtsam_id(lm_id)
                    valid_visual_factors.append((kf_gtsam_id, l_id, pt_2d))
                    active_lm_ids.add(l_id)

        # 5. 执行全局批量迭代优化
        try:
            start_time = time.time()
            self.active_values, new_factors_error = self._run_iterative_ba(
                base_graph, valid_visual_factors, self.active_kf_gtsam_ids, list(active_lm_ids), max_iters=2)
            print(f"【Backend Timer】: Initial optimization took {(time.time() - start_time) * 1000:.3f} ms.")
        except Exception as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!! INITIALIZATION FAILED !!!!!!!!!!!!!!")
            print(f"ERROR: {e}")
            return

        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        latest_gtsam_id = self.next_gtsam_kf_id - 1
        if latest_bias is not None:
            self.latest_bias = latest_bias
            
        print("【Backend】: Initial graph optimization complete.")
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        # 7. 后处理：边缘化第 0 帧
        oldest_gtsam_id = self.active_kf_gtsam_ids.pop(0)
        keys_to_marg_list = [X(oldest_gtsam_id), V(oldest_gtsam_id), B(oldest_gtsam_id)]
        keys_to_marg = gtsam.KeyVector()
        for k in keys_to_marg_list: keys_to_marg.append(k)

        try:
            # 基础图不含路标点，直接边缘化安全无虞
            self.marg_factor = gtsam.marginalizeOut(base_graph, self.active_values, keys_to_marg)
        except Exception as e:
            print(f"【Backend Init Error】: Marginalization failed: {e}")
            
        for k in keys_to_marg_list:
            if self.active_values.exists(k):
                self.active_values.erase(k)

    def window_optimize(self, active_kfs, window_imu_factors, window_landmarks,
                             window_visual_factors, new_kf_initial_guess, is_stationary):
        new_keyframe = active_kfs[-1]
        kf_gtsam_id = self._get_kf_gtsam_id(new_keyframe.get_id())
        T_wb_guess, vel_guess, bias_guess = new_kf_initial_guess

        # 1. 向 Active Values 中插入新状态，同时插入新的3D点
        if not self.active_values.exists(X(kf_gtsam_id)):
            self.active_values.insert(X(kf_gtsam_id), T_wb_guess)
            self.active_values.insert(V(kf_gtsam_id), vel_guess)
            self.active_values.insert(B(kf_gtsam_id), bias_guess)
            self.active_kf_gtsam_ids.append(kf_gtsam_id)

        for lm_id, lm_3d_pos in window_landmarks.items():
            if np.isnan(lm_3d_pos).any() or np.isinf(lm_3d_pos).any():
                continue
            lm_gtsam_id = self._get_lm_gtsam_id(lm_id)
            if lm_gtsam_id not in self.landmark_states:
                self.landmark_states[lm_gtsam_id] = np.array(lm_3d_pos)

        # 2. 从零构建当前的局部基础 FactorGraph
        base_graph = gtsam.NonlinearFactorGraph()

        # 2-1. 压入边缘化因子
        if self.marg_factor is not None:
            base_graph.push_back(self.marg_factor)

        # 2-2. 压入初始先验
        if X(0) in [X(id) for id in self.active_kf_gtsam_ids]:
            # self.init_priors 是初始化时存下来的绝对先验 (仅为了稳住0帧)
            for i in range(self.init_priors.size()):
                base_graph.push_back(self.init_priors.at(i))

        # 2-3. 压入IMU预积分因子
        for imu_data in window_imu_factors:
            id1, id2 = self._get_kf_gtsam_id(imu_data['kf_id1']), self._get_kf_gtsam_id(imu_data['kf_id2'])
            base_graph.push_back(gtsam.CombinedImuFactor(
                X(id1), V(id1), X(id2), V(id2), B(id1), B(id2), imu_data['pim']))

        # 2-4. 收集视觉因子
        valid_visual_factors = []
        active_lm_ids = set()

        for kf_id, lm_id, pt_2d in window_visual_factors:
            k_id = self._get_kf_gtsam_id(kf_id)
            if not self.active_values.exists(X(k_id)): continue
            l_id = self._get_lm_gtsam_id(lm_id)

            if l_id in self.landmark_states:
                # 简单手性预检
                depth = self._compute_landmark_depth(self.landmark_states[l_id], self.active_values.atPose3(X(k_id)))
                if depth < 0.1: continue

                valid_visual_factors.append((k_id, l_id, pt_2d))
                active_lm_ids.add(l_id)

        # ZUPT 逻辑
        if is_stationary:
            self.stationary_kf_gtsam_ids.add(kf_gtsam_id)
        zero_velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.03)
        for active_id in self.active_kf_gtsam_ids:
            if active_id in self.stationary_kf_gtsam_ids:
                base_graph.add(gtsam.PriorFactorVector(V(active_id), np.zeros(3), zero_velocity_noise))

        # 3. 执行局部 LM 迭代优化
        try:
            start_time = time.time()
            self.active_values, new_factors_error = self._run_iterative_ba(
                base_graph, valid_visual_factors, self.active_kf_gtsam_ids, list(active_lm_ids), max_iters=2)
            print(f"【Backend】: Optimization took {(time.time() - start_time) * 1000:.2f} ms")
        except RuntimeError as e:
            print(f"!!!!!!!!!! OPTIMIZATION FAILED !!!!!!!!!!!!!!\nERROR: {e}")
            return

        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        latest_gtsam_id = self.next_gtsam_kf_id - 1
        if latest_bias is not None: self.latest_bias = latest_bias

        print(f"【Backend】优化误差统计: 本轮初始视觉全局误差 = {new_factors_error:.4f}")
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        # 4. 滑动窗口边缘化
        oldest_gtsam_id = self.active_kf_gtsam_ids.pop(0)
        keys_to_marg_list = [X(oldest_gtsam_id), V(oldest_gtsam_id), B(oldest_gtsam_id)]

        # 边缘化时，利用收敛的位姿算出最后一次局部约束
        marg_pose_ids = self.active_kf_gtsam_ids + [oldest_gtsam_id]
        kf_states_dict = {kf: self.active_values.atPose3(X(kf)) for kf in marg_pose_ids}
        lm_states_dict = {lm: self.landmark_states[lm] for lm in active_lm_ids if lm in self.landmark_states}
        
        final_H_marg, final_b_marg, _, _, _, final_f_err, valid_lm_ids_marg = build_structureless_hessian(
            kf_states_dict, lm_states_dict, valid_visual_factors, self.K, self.body_T_cam, self.visual_noise_sigma)

        symbols = gtsam.KeyVector()
        for kf_id in marg_pose_ids:
            symbols.append(X(kf_id))
            
        final_graph_for_marg = gtsam.NonlinearFactorGraph(base_graph)
        if len(valid_lm_ids_marg) > 0:
            final_graph_for_marg.push_back(
                self._create_custom_hessian_factor(
                    symbols, self.active_values, final_H_marg, final_b_marg, final_f_err))

        keys_to_marg = gtsam.KeyVector()
        for k in keys_to_marg_list: keys_to_marg.append(k)

        try:
            self.marg_factor = gtsam.marginalizeOut(final_graph_for_marg, self.active_values, keys_to_marg)
            print(f"【Backend】: Marginalized old frame X({oldest_gtsam_id}) successfully.")
        except Exception as e:
            print(f"【Backend】: Marginalization Error: {e}")

        for k in keys_to_marg_list:
            if self.active_values.exists(k): self.active_values.erase(k)
        if oldest_gtsam_id in self.stationary_kf_gtsam_ids:
            self.stationary_kf_gtsam_ids.remove(oldest_gtsam_id)

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