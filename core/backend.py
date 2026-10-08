import time
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B, L

from utils.debug import Debugger

class Backend:
    def __init__(self, global_central_map, config, imu_processor):
        self.global_central_map = global_central_map
        self.config = config

        # 滑窗数据结构
        self.window_size = config.get('window_size', 10)
        self.active_values = gtsam.Values()
        self.active_frame_gtsam_ids = []
        self.marg_factor = None

        # 初始化先验因子缓存
        self.init_priors = gtsam.NonlinearFactorGraph()

        # 鲁棒因子
        self.visual_noise_sigma = config.get('visual_noise_sigma', 2.0)
        self.visual_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.visual_noise_sigma)
        self.visual_robust_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), self.visual_noise)
        self.max_reprojection_error = config.get('optimization_max_reprojection_error', 60.0)

        # 状态与id管理
        self.frame_id_to_gtsam_id = {}
        self.landmark_id_to_gtsam_id = {}
        self.next_gtsam_frame_id = 0

        # 获取相机内、外参
        cam_intrinsics = np.asarray(self.config.get('cam_intrinsics')).reshape(3, 3)
        self.K = gtsam.Cal3_S2(cam_intrinsics[0, 0], cam_intrinsics[1, 1], 0, 
                               cam_intrinsics[0, 2], cam_intrinsics[1, 2])
        self.fx = float(cam_intrinsics[0, 0])
        self.fy = float(cam_intrinsics[1, 1])
        self.cx = float(cam_intrinsics[0, 2])
        self.cy = float(cam_intrinsics[1, 2])

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

        # 用于记录历史静止帧，防止 ZUPT 断层跳变
        self.stationary_frame_gtsam_ids = set()

    # 帧 id 映射到图的 id
    def _get_frame_gtsam_id(self, frame_id):
        if frame_id not in self.frame_id_to_gtsam_id:
            self.frame_id_to_gtsam_id[frame_id] = self.next_gtsam_frame_id
            self.next_gtsam_frame_id += 1
        return self.frame_id_to_gtsam_id[frame_id]

    def _frame_id_for_gtsam_id(self, gtsam_id):
        for frame_id, mapped_id in self.frame_id_to_gtsam_id.items():
            if mapped_id == gtsam_id:
                return frame_id
        return None

    # 路标点id映射到图的id
    def _get_lm_gtsam_id(self, lm_id):
        if lm_id not in self.landmark_id_to_gtsam_id:
            self.landmark_id_to_gtsam_id[lm_id] = lm_id
        return self.landmark_id_to_gtsam_id[lm_id]

    def get_optimized_state(self, frame_id):
        gtsam_id = self.frame_id_to_gtsam_id.get(frame_id)
        if gtsam_id is None:
            return None, None, None
        try:
            if not (
                self.active_values.exists(X(gtsam_id))
                and self.active_values.exists(V(gtsam_id))
                and self.active_values.exists(B(gtsam_id))
            ):
                return None, None, None
            return (
                self.active_values.atPose3(X(gtsam_id)),
                self.active_values.atVector(V(gtsam_id)),
                self.active_values.atConstantBias(B(gtsam_id)),
            )
        except Exception as e:
            print(f"[Error][Backend] Failed to retrieve state of frame {frame_id}: {e}")
            return None, None, None

    def get_latest_optimized_state(self):
        if not self.active_frame_gtsam_ids:
            return None, None, None
        
        latest_gtsam_id = self.active_frame_gtsam_ids[-1]

        try:
            pose = self.active_values.atPose3(X(latest_gtsam_id))
            velocity = self.active_values.atVector(V(latest_gtsam_id))
            bias = self.active_values.atConstantBias(B(latest_gtsam_id))
            return pose, velocity, bias
        except Exception as e:
            print(f"[Error][Backend] Failed to retrieve latest state: {e}")
            return None, None, None

    def update_estimator_map(self, frame_window, landmarks):
        print("【Backend】: Syncing optimized results back to Estimator...")
        optimized_results = self.active_values

        # 1. 更新帧位姿
        for frame in frame_window:
            gtsam_id = self.frame_id_to_gtsam_id.get(frame.get_id())
            if gtsam_id is not None and optimized_results.exists(X(gtsam_id)):
                pose_w_b = optimized_results.atPose3(X(gtsam_id))
                frame.set_global_pose(pose_w_b.matrix())

        # 2. 从显式 Point3 变量回写路标
        for lm_id, landmark_obj in landmarks.items():
            lm_gtsam_id = self.landmark_id_to_gtsam_id.get(lm_id)
            if lm_gtsam_id is None:
                continue
            point_key = L(lm_gtsam_id)
            if not optimized_results.exists(point_key):
                continue
            landmark_obj.set_triangulated(np.asarray(optimized_results.atPoint3(point_key), dtype=float).reshape(3))

    def remove_stale_landmarks(self, unhealty_lm_ids, unhealty_lm_ids_depth, 
                                unhealty_lm_ids_reproj, oldest_frame_id_in_window):
        print(f"【Backend】: 接收到移除 {len(unhealty_lm_ids)} 个陈旧路标点的指令。")
        if not unhealty_lm_ids:
            return
        
        # 只删除ID映射，阻止这些landmark再次被添加到图中
        for lm_id in unhealty_lm_ids:
            if lm_id in self.landmark_id_to_gtsam_id:
                del self.landmark_id_to_gtsam_id[lm_id]
                print(f"【Backend】: 已移除 landmark {lm_id} 的ID映射")

    def _point2(self, pt):
        arr = np.asarray(pt, dtype=float).reshape(-1)
        return gtsam.Point2(float(arr[0]), float(arr[1]))

    def _landmark_position(self, lm_id, fallback):
        point_key = L(self._get_lm_gtsam_id(lm_id))
        if self.active_values.exists(point_key):
            return np.asarray(self.active_values.atPoint3(point_key), dtype=float).reshape(3)
        return np.asarray(fallback, dtype=float).reshape(3)

    def _camera_depth(self, position, gtsam_id):
        if not self.active_values.exists(X(gtsam_id)):
            return None
        T_wc = self.active_values.atPose3(X(gtsam_id)).matrix() @ self.T_bc
        p_c = T_wc[:3, :3].T @ (np.asarray(position, dtype=float).reshape(3) - T_wc[:3, 3])
        return float(p_c[2])

    def _reprojection_error_px(self, position, gtsam_id, pt):
        if not self.active_values.exists(X(gtsam_id)):
            return None
        T_wc = self.active_values.atPose3(X(gtsam_id)).matrix() @ self.T_bc
        p_c = T_wc[:3, :3].T @ (np.asarray(position, dtype=float).reshape(3) - T_wc[:3, 3])
        if p_c[2] <= 1e-8:
            return None
        u = self.fx * p_c[0] / p_c[2] + self.cx
        v = self.fy * p_c[1] / p_c[2] + self.cy
        measured = np.asarray(pt, dtype=float).reshape(-1)
        return float(np.hypot(u - measured[0], v - measured[1]))

    def _select_visual_groups(self, landmark_positions, visual_factors, landmark_hosts=None):
        """保留至少两个正深度、重投影未超阈值的观测。"""
        # 筛选有效观测，并组装成观测组，包括正深度检查、重投影误差检查、host检查
        grouped = {}
        hosts = landmark_hosts or {}
        for frame_id, lm_id, pt in visual_factors:
            if lm_id not in landmark_positions or landmark_positions[lm_id] is None:
                continue
            arr = np.asarray(pt, dtype=float).reshape(-1)
            if arr.size < 2:
                continue
            group = grouped.setdefault(lm_id, {
                'position': np.asarray(landmark_positions[lm_id], dtype=float).reshape(3),
                'host': hosts.get(lm_id),
                'observations': []
            })
            group['observations'].append((frame_id, arr[:2].copy()))

        selected = {}
        for lm_id, group in grouped.items():
            if group['host'] is None and group['observations']:
                group['host'] = min(frame_id for frame_id, _ in group['observations'])
            valid_obs = []
            position = self._landmark_position(lm_id, group['position'])
            for frame_id, pt in group['observations']:
                gtsam_id = self.frame_id_to_gtsam_id.get(frame_id)
                if gtsam_id is None or not self.active_values.exists(X(gtsam_id)):
                    continue
                depth = self._camera_depth(position, gtsam_id)
                if depth is None or depth <= 0.0:
                    continue
                error = self._reprojection_error_px(position, gtsam_id, pt)
                if error is None or error > self.max_reprojection_error:
                    continue
                valid_obs.append((frame_id, pt))
            if len(valid_obs) < 2:
                continue
            group['observations'] = valid_obs
            group['position'] = position
            selected[lm_id] = group
        return selected

    def _append_imu_factors(self, graph, imu_factors, start_frame_id=None):
        consumed = []
        for imu_data in imu_factors:
            id1 = imu_data['frame_id1']
            id2 = imu_data['frame_id2']
            if start_frame_id is not None and id1 != start_frame_id:
                continue
            g1 = self.frame_id_to_gtsam_id.get(id1)
            g2 = self.frame_id_to_gtsam_id.get(id2)
            if g1 is None or g2 is None:
                continue
            needed = (X(g1), V(g1), B(g1), X(g2), V(g2), B(g2))
            if any(not self.active_values.exists(key) for key in needed):
                continue
            graph.add(gtsam.CombinedImuFactor(
                X(g1), V(g1), X(g2), V(g2), B(g1), B(g2), imu_data['pim']))
            consumed.append((id1, id2))
        return consumed

    def _append_projection_factors(self, graph, groups, host_frame_id=None, insert_values=True):
        used = []
        for lm_id, group in groups.items():
            if host_frame_id is not None and int(group['host']) != int(host_frame_id):
                continue
            point_key = L(self._get_lm_gtsam_id(lm_id))
            if not self.active_values.exists(point_key):
                if not insert_values:
                    continue
                position = np.asarray(group['position'], dtype=float).reshape(3)
                self.active_values.insert(
                    point_key, gtsam.Point3(float(position[0]), float(position[1]), float(position[2])))
            added = 0
            for frame_id, pt in group['observations']:
                pose_id = self.frame_id_to_gtsam_id.get(frame_id)
                if pose_id is None or not self.active_values.exists(X(pose_id)):
                    continue
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    self._point2(pt), self.visual_robust_noise, X(pose_id), point_key, self.K, self.body_T_cam))
                added += 1
            if added > 0:
                used.append(lm_id)
        return used

    def _append_zupt(self, graph, gtsam_ids):
        noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.03)
        for gtsam_id in gtsam_ids:
            if gtsam_id in self.stationary_frame_gtsam_ids and self.active_values.exists(V(gtsam_id)):
                graph.add(gtsam.PriorFactorVector(V(gtsam_id), np.zeros(3), noise))

    def _build_optimization_graph(self, imu_factors, visual_groups):
        graph = gtsam.NonlinearFactorGraph()
        # 压入边缘化因子
        if self.marg_factor is not None:
            graph.push_back(self.marg_factor)
        # 压入初始先验
        if 0 in self.active_frame_gtsam_ids and self.init_priors.size() > 0:
            graph.push_back(self.init_priors)
        # 压入IMU预积分因子
        self._append_imu_factors(graph, imu_factors)
        # 压入视觉因子
        self._append_projection_factors(graph, visual_groups, insert_values=True)
        # 压入ZUPT因子
        self._append_zupt(graph, self.active_frame_gtsam_ids)
        return graph

    def _graph_keys(self, graph):
        keys = set()
        for i in range(graph.size()):
            factor = graph.at(i)
            if factor is None:
                continue
            for key in factor.keys():
                keys.add(key)
        return keys

    def _erase_orphan_landmarks(self, graph):
        graph_keys = self._graph_keys(graph)
        for lm_gtsam_id in self.landmark_id_to_gtsam_id.values():
            point_key = L(lm_gtsam_id)
            if self.active_values.exists(point_key) and point_key not in graph_keys:
                self.active_values.erase(point_key)

    def marginalize_oldest(self, imu_factors, visual_groups):
        """只把最老帧、第一段 IMU，以及 host 属于该帧的整条视觉轨迹收进先验。"""
        if not self.active_frame_gtsam_ids:
            return None
        gtsam_id = self.active_frame_gtsam_ids[0]
        frame_id = self._frame_id_for_gtsam_id(gtsam_id)
        if frame_id is None:
            return None

        # 1. 构建边缘化图
        marg_graph = gtsam.NonlinearFactorGraph()
        if self.marg_factor is not None:
            marg_graph.push_back(self.marg_factor)
        if gtsam_id == 0 and self.init_priors.size() > 0:
            marg_graph.push_back(self.init_priors)

        # 添加和边缘化帧相关的IMU因子
        consumed_imu = self._append_imu_factors(marg_graph, imu_factors, start_frame_id=frame_id) 
        # 添加和边缘化帧相关的ZUPT因子
        self._append_zupt(marg_graph, [gtsam_id]) 
        # 添加和边缘化帧相关的视觉因子
        marg_landmarks = self._append_projection_factors(
            marg_graph, visual_groups, host_frame_id=frame_id, insert_values=False) 

        # 收集边缘化图中的和边缘化帧相关的因子（边缘化帧自己的X/V/B和与其相连的路标点因子）
        graph_keys = self._graph_keys(marg_graph)
        frame_keys = [X(gtsam_id), V(gtsam_id), B(gtsam_id)]
        if any(key not in graph_keys for key in frame_keys):
            print(f"【Backend】: Frame {frame_id} is missing pose, velocity, or bias factors. Skip marginalization.")
            return None

        # 和边缘化帧相关路标点因子
        keys_to_marg_list = []
        for lm_id in marg_landmarks:
            point_key = L(self._get_lm_gtsam_id(lm_id))
            if point_key in graph_keys:
                keys_to_marg_list.append(point_key)
        
        # 边缘化帧自己的X/V/B
        keys_to_marg_list.extend(frame_keys)

        keys_to_marg = gtsam.KeyVector()
        for key in keys_to_marg_list:
            keys_to_marg.append(key)

        try:
            new_marg_factor = gtsam.marginalizeOut(marg_graph, self.active_values, keys_to_marg)
        except Exception as e:
            print(f"【Backend】: Marginalization Error: {e}")
            return None

        self.marg_factor = new_marg_factor
        self.active_frame_gtsam_ids.remove(gtsam_id)
        for key in keys_to_marg_list:
            if self.active_values.exists(key):
                self.active_values.erase(key)
        self.stationary_frame_gtsam_ids.discard(gtsam_id)

        print(
            f"【Backend】: Marginalized oldest frame {frame_id} X({gtsam_id}) "
            f"and {len(marg_landmarks)} host landmarks."
        )
        return {
            'success': True,
            'mode': 'MARGIN_OLD',
            'marginalized_frame_id': frame_id,
            'marginalized_landmark_ids': list(marg_landmarks),
            'consumed_imu_factor_ids': consumed_imu,
        }

    def marginalize_second_newest(self, frame_id):
        """只从旧先验中消除次新帧。不吸收 IMU 和视觉因子。"""
        if len(self.active_frame_gtsam_ids) < 2:
            print("【Backend】: Not enough active states for MARGIN_SECOND_NEW.")
            return {'success': False}
        gtsam_id = self.active_frame_gtsam_ids[-2]
        active_frame_id = self._frame_id_for_gtsam_id(gtsam_id)
        if active_frame_id != frame_id:
            print(
                f"【Backend】: Frame {frame_id} is not the second-newest active state "
                f"({active_frame_id}). Skip marginalization."
            )
            return {'success': False}

        frame_keys = [X(gtsam_id), V(gtsam_id), B(gtsam_id)]
        marg_graph = gtsam.NonlinearFactorGraph()
        # 这里的marg_factor是上一轮边缘化时生成的边缘化因子
        if self.marg_factor is not None:
            marg_graph.push_back(self.marg_factor)
        prior_keys = self._graph_keys(marg_graph)
        keys_to_marg_list = [key for key in frame_keys if key in prior_keys]

        candidate_prior = self.marg_factor
        if keys_to_marg_list:
            keys_to_marg = gtsam.KeyVector()
            for key in keys_to_marg_list:
                keys_to_marg.append(key)
            try:
                candidate_prior = gtsam.marginalizeOut(marg_graph, self.active_values, keys_to_marg)
            except Exception as e:
                print(f"【Backend】: Second-newest marginalization error: {e}")
                return {'success': False}

        if candidate_prior is not None:
            committed_graph = gtsam.NonlinearFactorGraph()
            committed_graph.push_back(candidate_prior)
            committed_keys = self._graph_keys(committed_graph)
            if any(key in committed_keys for key in frame_keys):
                print(f"【Backend】: Prior still references frame {frame_id} after marginalization.")
                return {'success': False}

        self.marg_factor = candidate_prior
        self.active_frame_gtsam_ids.remove(gtsam_id)
        for key in frame_keys:
            if self.active_values.exists(key):
                self.active_values.erase(key)
        self.stationary_frame_gtsam_ids.discard(gtsam_id)

        print(f"【Backend】: Marginalized second-newest frame {frame_id} X({gtsam_id}).")
        return {
            'success': True,
            'mode': 'MARGIN_SECOND_NEW',
            'marginalized_frame_id': frame_id,
            'marginalized_gtsam_id': gtsam_id,
            'marginalized_landmark_ids': [],
        }

    def initialize_optimize(self, initial_keyframes, initial_imu_factors, initial_landmarks, initial_velocities, initial_bias):
        print("【Backend】: Initializing optimize...")

        # 确保初始化时状态机是干净的
        self.active_values.clear()
        self.active_frame_gtsam_ids.clear()
        self.marg_factor = None
        self.init_priors = gtsam.NonlinearFactorGraph()
        self.stationary_frame_gtsam_ids.clear()
        
        # ---------------------------------------------------------
        # 1. 插入初始帧状态 (Pose, Vel, Bias) 并添加强先验
        # ---------------------------------------------------------
        for i, kf in enumerate(initial_keyframes):
            frame_gtsam_id = self._get_frame_gtsam_id(kf.get_id())
            self.active_frame_gtsam_ids.append(frame_gtsam_id)

            T_wb = gtsam.Pose3(kf.get_global_pose())
            velocity = initial_velocities[i*3 : i*3+3]
            bias = initial_bias 

            self.active_values.insert(X(frame_gtsam_id), T_wb)
            self.active_values.insert(V(frame_gtsam_id), velocity)
            self.active_values.insert(B(frame_gtsam_id), bias)

            # 为第一帧添加强先验
            if frame_gtsam_id == 0:
                prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*3 + [1e-2]*3))
                prior_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2e-2] * 3))
                prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1]*3 + [1e-2]*3))

                f_pose = gtsam.PriorFactorPose3(X(0), T_wb, prior_pose_noise)
                f_vel = gtsam.PriorFactorVector(V(0), velocity, prior_vel_noise)
                f_bias = gtsam.PriorFactorConstantBias(B(0), bias, prior_bias_noise)

                self.init_priors.add(f_pose)
                self.init_priors.add(f_vel)
                self.init_priors.add(f_bias)

        # ---------------------------------------------------------
        # 2. 整理 IMU 与视觉观测，构建完整优化图
        # ---------------------------------------------------------
        imu_factors = []
        for factor_data in initial_imu_factors:
            start_frame = next(frame for frame in initial_keyframes if frame.get_timestamp() == factor_data['start_timestamp'])
            end_frame = next(frame for frame in initial_keyframes if frame.get_timestamp() == factor_data['end_timestamp'])
            imu_factors.append({
                'frame_id1': start_frame.get_id(),
                'frame_id2': end_frame.get_id(),
                'pim': factor_data['imu_preintegration'],
            })

        visual_factors = []
        landmark_hosts = {}
        for kf in initial_keyframes:
            for lm_id, pt_2d in zip(kf.get_visual_feature_ids(), kf.get_visual_features()):
                if lm_id not in initial_landmarks or initial_landmarks[lm_id] is None:
                    continue
                visual_factors.append((kf.get_id(), lm_id, pt_2d))
                landmark_hosts.setdefault(lm_id, kf.get_id())

        visual_groups = self._select_visual_groups(initial_landmarks, visual_factors, landmark_hosts)
        graph = self._build_optimization_graph(imu_factors, visual_groups)
        print(
            f"【Backend】: Initializing graph with {graph.size()} factors, "
            f"{self.active_values.size()} variables, {len(visual_groups)} landmarks."
        )

        # ---------------------------------------------------------
        # 3. 执行全局批量优化
        # ---------------------------------------------------------
        try:
            start_time = time.time()
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, self.active_values, params)
            self.active_values = optimizer.optimize()
            end_time = time.time()
            print(f"【Backend Timer】: Initial optimization took { (end_time - start_time) * 1000:.3f} ms.")
        except Exception as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!! INITIALIZATION FAILED !!!!!!!!!!!!!!")
            print(f"ERROR: {e}")
            return None

        # ---------------------------------------------------------
        # 4. 更新并记录状态
        # ---------------------------------------------------------
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        latest_gtsam_id = self.next_gtsam_frame_id - 1
        if latest_pose is None or latest_vel is None or latest_bias is None:
            print("【Backend】Critical: Initial optimization succeeded but state retrieval failed.")
            return None
        self.latest_bias = latest_bias

        new_factors_error = self._log_optimization_error(graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)

        # ---------------------------------------------------------
        # 5. 用独立边缘化图消除最老帧及其 host 路标
        # ---------------------------------------------------------
        return self.marginalize_oldest(imu_factors, visual_groups)


    def optimize_window(self, active_frames, window_imu_factors, window_landmarks,
                        window_visual_factors, new_frame_initial_guess, is_stationary,
                        landmark_hosts=None):
        new_frame = active_frames[-1]
        frame_gtsam_id = self._get_frame_gtsam_id(new_frame.get_id())
        T_wb_guess, vel_guess, bias_guess = new_frame_initial_guess

        # ---------------------------------------------------------
        # 1. 插入新帧状态。已有 key 不重新编号。
        # ---------------------------------------------------------
        if not self.active_values.exists(X(frame_gtsam_id)):
            self.active_values.insert(X(frame_gtsam_id), T_wb_guess)
            self.active_values.insert(V(frame_gtsam_id), vel_guess)
            self.active_values.insert(B(frame_gtsam_id), bias_guess)
            self.active_frame_gtsam_ids.append(frame_gtsam_id)

        if is_stationary:
            self.stationary_frame_gtsam_ids.add(frame_gtsam_id)

        # ---------------------------------------------------------
        # 2. 构建完整优化图：先验 + 全部 IMU + 全部有效重投影 + ZUPT
        # ---------------------------------------------------------
        # 收集所有健康的 3D 点和 2D 观测
        visual_groups = self._select_visual_groups(window_landmarks, window_visual_factors, landmark_hosts)
        # 构建完整优化图：先验 + 全部 IMU + 全部有效重投影 + ZUPT
        current_graph = self._build_optimization_graph(window_imu_factors, visual_groups)
        self._erase_orphan_landmarks(current_graph)
        print(
            f"【Backend】: Window graph has {current_graph.size()} factors "
            f"and {len(visual_groups)} landmarks."
        )

        # ---------------------------------------------------------
        # 3. 执行局部 LM 优化
        # ---------------------------------------------------------
        try:
            start_time = time.time()
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(current_graph, self.active_values, params)
            self.active_values = optimizer.optimize()
            print(f"【Backend】: Optimization took {(time.time() - start_time) * 1000:.2f} ms")
        except Exception as e:
            print(f"!!!!!!!!!! OPTIMIZATION FAILED !!!!!!!!!!!!!!\nERROR: {e}")
            return {'success': False}

        # ---------------------------------------------------------
        # 4. 更新最新状态及记录状态
        # ---------------------------------------------------------
        latest_pose, latest_vel, latest_bias = self.get_latest_optimized_state()
        latest_gtsam_id = self.next_gtsam_frame_id - 1
        if latest_bias is not None:
            self.latest_bias = latest_bias

        if latest_pose is None:
             print("【Backend】Critical: Optimization succeeded but state retrieval failed.")
             return {'success': False}

        new_factors_error = self._log_optimization_error(current_graph)
        self._log_state_and_errors(latest_gtsam_id, latest_pose, latest_vel, latest_bias, new_factors_error)
        return {
            'success': True,
            'latest_frame_id': new_frame.get_id(),
            'visual_groups': visual_groups,
        }


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