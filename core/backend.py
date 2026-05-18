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

        # 状态与id管理
        self.kf_id_to_gtsam_id = {}
        self.landmark_id_to_gtsam_id = {}
        self.next_gtsam_kf_id = 0

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

        # 用于彻底拦截已经被化为先验的路标点，防止前端诈尸
        self.marginalized_landmarks = set()

        # 用于记录历史静止帧，防止 ZUPT 断层跳变
        self.stationary_kf_gtsam_ids = set()

        # 用于暂存当前的 Smart Factors，以便提取 3D 点还给前端
        self.current_smart_factors = {}

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

        # 1. 更新关键帧位姿
        for kf in keyframe_window:
            gtsam_id = self.kf_id_to_gtsam_id.get(kf.get_id())
            if gtsam_id is not None and optimized_results.exists(X(gtsam_id)):
                pose_w_b = optimized_results.atPose3(X(gtsam_id))
                kf.set_global_pose(pose_w_b.matrix())

        # 2. 从 Smart Factor 反向提取路标点 3D 坐标还给前端
        for lm_id, smart_factor in self.current_smart_factors.items():
            if lm_id not in landmarks:
                continue
            landmark_obj = landmarks[lm_id]
            try:
                p3_status = smart_factor.point(optimized_results)
                if p3_status is not None:
                    try:
                        optimized_position = np.array([p3_status.x(), p3_status.y(), p3_status.z()])
                        landmark_obj.set_triangulated(optimized_position)
                    except AttributeError:
                        pass
            except Exception:
                pass

    def remove_stale_landmarks(self, unhealty_lm_ids, unhealty_lm_ids_depth, 
                                unhealty_lm_ids_reproj, oldest_kf_id_in_window):
        print(f"【Backend】: 接收到移除 {len(unhealty_lm_ids)} 个陈旧路标点的指令。")
        if not unhealty_lm_ids:
            return
        
        # 只删除ID映射，阻止这些landmark再次被添加到图中
        for lm_id in unhealty_lm_ids:
            if lm_id in self.landmark_id_to_gtsam_id:
                del self.landmark_id_to_gtsam_id[lm_id]
                print(f"【Backend】: 已移除 landmark {lm_id} 的ID映射")

    def _create_smart_params(self):
        """统一配置 SmartFactor 的参数，防退化配置"""
        smart_params = gtsam.SmartProjectionParams()
        smart_params.setLinearizationMode(gtsam.LinearizationMode.HESSIAN) 
        try:
            smart_params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)
        except AttributeError:
            print("[Backend Warning]: GTSAM version might be old, DegeneracyMode not found.")
        return smart_params

    def initialize_optimize(self, initial_keyframes, initial_imu_factors, initial_landmarks, initial_velocities, initial_bias):
        print("【Backend】: Initializing optimize...")

        graph = gtsam.NonlinearFactorGraph()

        # 确保初始化时状态机是干净的
        self.active_values.clear()
        self.active_kf_gtsam_ids.clear()
        self.marg_factor = None
        self.stationary_kf_gtsam_ids.clear()
        self.current_smart_factors.clear()
        
        # ---------------------------------------------------------
        # 1. 插入初始帧状态 (Pose, Vel, Bias) 并添加强先验
        # ---------------------------------------------------------
        for i, kf in enumerate(initial_keyframes):
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            self.active_kf_gtsam_ids.append(kf_gtsam_id)

            T_wb = gtsam.Pose3(kf.get_global_pose())
            velocity = initial_velocities[i*3 : i*3+3]
            bias = initial_bias 

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

                self.init_priors.add(f_pose)
                self.init_priors.add(f_vel)
                self.init_priors.add(f_bias)

        # ---------------------------------------------------------
        # 2. 压入所有 IMU 预积分因子
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
        # 3. 收集并压入视觉因子 (Smart Factor 无结构化处理)
        # ---------------------------------------------------------
        smart_params = self._create_smart_params()
        smart_observations = {}
        
        for kf in initial_keyframes:
            kf_gtsam_id = self._get_kf_gtsam_id(kf.get_id())
            for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                # 仅处理在 initial_landmarks 里的有效点
                if lm_id in initial_landmarks:
                    if lm_id not in smart_observations:
                        smart_observations[lm_id] = []
                    smart_observations[lm_id].append((kf_gtsam_id, pt_2d))

        for lm_id, obs_list in smart_observations.items():
            if len(obs_list) < 2:
                continue
            # 初始化智能因子 (不带 L 变量)
            smart_factor = gtsam.SmartProjectionPose3Factor(
                self.visual_noise, self.K, self.body_T_cam, smart_params)
            
            for kf_gtsam_id, pt_2d in obs_list:
                smart_factor.add(pt_2d, X(kf_gtsam_id))
            
            graph.push_back(smart_factor)

        # ---------------------------------------------------------
        # 4. 执行全局批量优化
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
        # 5. 更新并记录状态
        # ---------------------------------------------------------
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        latest_gtsam_id = self.next_gtsam_kf_id - 1
        if latest_bias is not None:
            self.latest_bias = latest_bias

        new_factors_error = self._log_optimization_error(graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        # ---------------------------------------------------------
        # 6. 极端情况的初始边缘化 (只包含 X, V, B)
        # ---------------------------------------------------------
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
        old_keyframe = active_kfs[0]
        new_keyframe = active_kfs[-1]
        kf_gtsam_id = self._get_kf_gtsam_id(new_keyframe.get_id())
        T_wb_guess, vel_guess, bias_guess = new_kf_initial_guess

        # 防火墙：过滤历史点
        window_landmarks = {k: v for k, v in window_landmarks.items() if k not in self.marginalized_landmarks}
        window_visual_factors = [f for f in window_visual_factors if f[1] not in self.marginalized_landmarks]

        # ---------------------------------------------------------
        # 1. 插入新帧状态
        # ---------------------------------------------------------
        if not self.active_values.exists(X(kf_gtsam_id)):
            self.active_values.insert(X(kf_gtsam_id), T_wb_guess)
            self.active_values.insert(V(kf_gtsam_id), vel_guess)
            self.active_values.insert(B(kf_gtsam_id), bias_guess)
            self.active_kf_gtsam_ids.append(kf_gtsam_id)

        # ---------------------------------------------------------
        # 2. 构建当前的局部 FactorGraph
        # ---------------------------------------------------------
        current_graph = gtsam.NonlinearFactorGraph()

        # [A] 压入边缘化因子和初始先验
        if self.marg_factor is not None:
            current_graph.push_back(self.marg_factor)
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

        # [C] 收集并压入视觉因子 (Smart Factor 无结构化处理)
        smart_params = self._create_smart_params()
        smart_observations = {}
        
        for kf_id, lm_id, pt_2d in window_visual_factors:
            k_id = self._get_kf_gtsam_id(kf_id)
            if not self.active_values.exists(X(k_id)):
                continue
                
            if lm_id not in smart_observations:
                smart_observations[lm_id] = []
            smart_observations[lm_id].append((k_id, pt_2d))

        self.current_smart_factors.clear()

        for lm_id, obs_list in smart_observations.items():
            if len(obs_list) < 2:
                continue

            smart_factor = gtsam.SmartProjectionPose3Factor(
                self.visual_noise, self.K, self.body_T_cam, smart_params)
            
            for kf_gtsam_id, pt_2d in obs_list:
                smart_factor.add(pt_2d, X(kf_gtsam_id))
            
            current_graph.push_back(smart_factor)
            self.current_smart_factors[lm_id] = smart_factor

        # [D] 零速度更新 (ZUPT) - 维护历史静止状态，防止起飞跳变
        if is_stationary:
            self.stationary_kf_gtsam_ids.add(kf_gtsam_id)

        zero_velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.03)
        for active_id in self.active_kf_gtsam_ids:
            if active_id in self.stationary_kf_gtsam_ids:
                zero_velocity_prior = gtsam.PriorFactorVector(
                    V(active_id), np.zeros(3), zero_velocity_noise)
                current_graph.add(zero_velocity_prior)

        # ---------------------------------------------------------
        # 2.5 清理孤儿变量
        # ---------------------------------------------------------
        graph_keys = set()
        for i in range(current_graph.size()):
            factor = current_graph.at(i)
            if factor is not None:
                for key in factor.keys():
                    graph_keys.add(key)

        orphan_keys = [k for k in self.active_values.keys() if k not in graph_keys]
        if orphan_keys:
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
            print(f"!!!!!!!!!! OPTIMIZATION FAILED !!!!!!!!!!!!!!\nERROR: {e}")
            return

        # ---------------------------------------------------------
        # 4. 更新最新状态及记录状态
        # ---------------------------------------------------------
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        latest_gtsam_id = self.next_gtsam_kf_id - 1
        if latest_bias is not None:
            self.latest_bias = latest_bias

        if latest_pose is None:
             print("【Backend】Critical: Optimization succeeded but state retrieval failed.")
             return

        new_factors_error = self._log_optimization_error(current_graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        # ---------------------------------------------------------
        # 5. 滑动窗口边缘化 (绝对恒定的右下角)
        # ---------------------------------------------------------
        oldest_gtsam_id = self.active_kf_gtsam_ids.pop(0) 
        keys_to_marg_list = [X(oldest_gtsam_id), V(oldest_gtsam_id), B(oldest_gtsam_id)] 

        keys_to_marg = gtsam.KeyVector()
        for k in keys_to_marg_list:
            keys_to_marg.append(k)

        try:
            # 这里的 marginalizeOut 因为图里没有 Landmark，速度将极快！
            new_marg_factor = gtsam.marginalizeOut(current_graph, self.active_values, keys_to_marg)
            self.marg_factor = new_marg_factor 
            print(f"【Backend】: Marginalized old frame X({oldest_gtsam_id}) successfully.")
        except Exception as e:
            print(f"【Backend】: Marginalization Error: {e}")

        for k in keys_to_marg_list:
            if self.active_values.exists(k):
                self.active_values.erase(k)

        if oldest_gtsam_id in self.stationary_kf_gtsam_ids:
            self.stationary_kf_gtsam_ids.remove(oldest_gtsam_id)


    def _log_optimization_error(self, current_full_graph):
        try:
            optimized_result = self.active_values
            new_factors_error = current_full_graph.error(optimized_result)
            print(f"【Backend】优化误差统计: 本轮全局误差 = {new_factors_error:.4f}")
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