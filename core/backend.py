import time
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B

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
        self.visual_noise_sigma = config.get('visual_noise_sigma', 1.5)
        self.visual_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.visual_noise_sigma)
        robust_kernel = str(config.get('visual_robust_kernel', 'cauchy')).lower()
        if robust_kernel == 'huber':
            estimator = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
        elif robust_kernel == 'cauchy':
            estimator = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
        else:
            raise ValueError(f"Unsupported visual_robust_kernel: {robust_kernel}")
        self.visual_robust_noise = gtsam.noiseModel.Robust.Create(estimator, self.visual_noise)
        self.max_reprojection_error = config.get('optimization_max_reprojection_error', 60.0)

        # 状态与id管理
        self.frame_id_to_gtsam_id = {}
        self.landmark_id_to_gtsam_id = {}
        self.optimized_landmark_positions = {}
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
        # VINS-Mono does not add zero-velocity priors. Keep this only as an
        # explicitly enabled extension for non-VINS experiments.
        self.use_zupt = bool(self.config.get('use_zupt', False))
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

        # 2. 从后端路标变量回写地图。逆深度变量以 host 相机为基准，
        # 因而必须用优化后的 host 位姿恢复世界坐标。
        for lm_id, landmark_obj in landmarks.items():
            lm_gtsam_id = self.landmark_id_to_gtsam_id.get(lm_id)
            if lm_gtsam_id is None:
                continue
            position = self._inverse_depth_world_position(lm_id, landmark_obj)
            if position is None:
                position = self.optimized_landmark_positions.get(lm_id)
            if position is not None:
                landmark_obj.set_triangulated(np.asarray(position, dtype=float).reshape(3))

    def remove_stale_landmarks(self, unhealty_lm_ids, unhealty_lm_ids_depth, 
                                unhealty_lm_ids_reproj, oldest_frame_id_in_window):
        print(f"【Backend】: 接收到移除 {len(unhealty_lm_ids)} 个陈旧路标点的指令。")
        if not unhealty_lm_ids:
            return
        
        # 只删除ID映射，阻止这些landmark再次被添加到图中
        for lm_id in unhealty_lm_ids:
            if lm_id in self.landmark_id_to_gtsam_id:
                del self.landmark_id_to_gtsam_id[lm_id]
                self.optimized_landmark_positions.pop(lm_id, None)
                print(f"【Backend】: 已移除 landmark {lm_id} 的ID映射")

    def _point2(self, pt):
        arr = np.asarray(pt, dtype=float).reshape(-1)
        return gtsam.Point2(float(arr[0]), float(arr[1]))

    def _inverse_depth_key(self, landmark_gtsam_id):
        return gtsam.symbol('d', int(landmark_gtsam_id))

    def _landmark_position(self, lm_id, fallback):
        return np.asarray(
            self.optimized_landmark_positions.get(lm_id, fallback), dtype=float
        ).reshape(3)

    def _host_observation(self, group):
        host_frame_id = group.get('host')
        for frame_id, pixel in group.get('observations', []):
            if int(frame_id) == int(host_frame_id):
                return np.asarray(pixel, dtype=float).reshape(2)
        return None

    def _inverse_depth_from_group(self, group):
        host_id = self.frame_id_to_gtsam_id.get(group.get('host'))
        if host_id is None or not self.active_values.exists(X(host_id)):
            return None
        T_wc = self.active_values.atPose3(X(host_id)).matrix() @ self.T_bc
        position = np.asarray(group['position'], dtype=float).reshape(3)
        point_camera = T_wc[:3, :3].T @ (position - T_wc[:3, 3])
        if point_camera[2] <= 1e-8:
            return None
        return 1.0 / float(point_camera[2])

    def _inverse_depth_world_position(self, lm_id, landmark_obj):
        lm_gtsam_id = self.landmark_id_to_gtsam_id.get(lm_id)
        host_frame_id = getattr(landmark_obj, 'host_frame_id', None)
        if lm_gtsam_id is None or host_frame_id is None:
            return None
        depth_key = self._inverse_depth_key(lm_gtsam_id)
        host_id = self.frame_id_to_gtsam_id.get(host_frame_id)
        if (host_id is None or not self.active_values.exists(X(host_id))
                or not self.active_values.exists(depth_key)):
            return None
        rho = float(self.active_values.atDouble(depth_key))
        if rho <= 1e-9 or host_frame_id not in landmark_obj.observations:
            return None
        pixel = np.asarray(landmark_obj.observations[host_frame_id], dtype=float).reshape(-1)
        ray = np.array([
            (pixel[0] - self.cx) / self.fx,
            (pixel[1] - self.cy) / self.fy,
            1.0,
        ])
        point_camera = ray / rho
        T_wc = self.active_values.atPose3(X(host_id)).matrix() @ self.T_bc
        return T_wc[:3, :3] @ point_camera + T_wc[:3, 3]

    def _cache_optimized_inverse_depth_positions(self, groups):
        for lm_id, group in groups.items():
            lm_gtsam_id = self.landmark_id_to_gtsam_id.get(lm_id)
            host_frame_id = group.get('host')
            host_id = self.frame_id_to_gtsam_id.get(host_frame_id)
            if lm_gtsam_id is None or host_id is None:
                continue
            depth_key = self._inverse_depth_key(lm_gtsam_id)
            if (not self.active_values.exists(depth_key)
                    or not self.active_values.exists(X(host_id))):
                continue
            rho = float(self.active_values.atDouble(depth_key))
            host_pixel = self._host_observation(group)
            if rho <= 1e-9 or host_pixel is None:
                continue
            ray = np.array([
                (host_pixel[0] - self.cx) / self.fx,
                (host_pixel[1] - self.cy) / self.fy,
                1.0,
            ])
            T_wc = self.active_values.atPose3(X(host_id)).matrix() @ self.T_bc
            point_camera = ray / rho
            self.optimized_landmark_positions[lm_id] = (
                T_wc[:3, :3] @ point_camera + T_wc[:3, 3]
            )

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

    def _append_inverse_depth_factors(self, graph, groups, host_frame_id=None, insert_values=True):
        used = []
        # 遍历所有路标点
        # 每个 group 里有三样东西：
        # position：这个点当前的世界坐标
        # host：它的首观测帧
        # observations：[(frame_id, pixel), ...]，这个点在窗口里留下来的观测
        for lm_id, group in groups.items():
            group_host = group.get('host') # 获取所有观测帧中host帧的ID
            if group_host is None:
                continue

            # 如果host_frame_id不为空，并且host帧的ID不等于指定的host_frame_id，则跳过该路标点
            if host_frame_id is not None and int(group_host) != int(host_frame_id):
                continue

            # 获取host帧的ID
            host_pose_id = self.frame_id_to_gtsam_id.get(group_host)

            # 获取host帧的特征点观测像素坐标
            host_pixel = self._host_observation(group)
            if (host_pose_id is None or host_pixel is None
                    or not self.active_values.exists(X(host_pose_id))):
                continue

            # 获取路标点的逆深度因子键
            depth_key = self._inverse_depth_key(self._get_lm_gtsam_id(lm_id))

            # 如果当前路标点的逆深度因子键不存在，则插入逆深度值（第一次观测到该路标点）
            if not self.active_values.exists(depth_key):
                if not insert_values:
                    continue
                # 计算当前路标点的逆深度值（host帧相机坐标系下的逆深度值）
                rho = self._inverse_depth_from_group(group)
                if rho is None:
                    continue
                self.active_values.insert(depth_key, float(rho))

            # 遍历路标点的剩余所有非host观测帧，插入逆深度因子
            added = 0
            for frame_id, pixel in group['observations']:
                if int(frame_id) == int(group_host):
                    continue
                target_pose_id = self.frame_id_to_gtsam_id.get(frame_id)
                if target_pose_id is None or not self.active_values.exists(X(target_pose_id)):
                    continue
                graph.add(gtsam.InverseDepthFactor(
                    self._point2(pixel), self._point2(host_pixel),
                    self.visual_robust_noise,
                    X(host_pose_id), X(target_pose_id), depth_key,
                    self.K, self.body_T_cam))
                added += 1
            if added > 0:
                used.append(lm_id)

        return used

    def _append_zupt(self, graph, gtsam_ids):
        if not self.use_zupt:
            return
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
        self._append_inverse_depth_factors(graph, visual_groups, insert_values=True)
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
            depth_key = self._inverse_depth_key(lm_gtsam_id)
            if self.active_values.exists(depth_key) and depth_key not in graph_keys:
                self.active_values.erase(depth_key)

    def marginalize_oldest(self, imu_factors, visual_groups, observation_frames=None):
        """消除最老帧的 X/V/B。host 属于该帧的逆深度随整条轨迹一起消除。"""
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

        # 深度属于首观测帧。host 离开窗口时整条轨迹进入边缘化图，并消除旧逆深度。
        # 当前窗口中所有将边缘化帧作为host帧的逆深度因子
        hosted = self._append_inverse_depth_factors(
            marg_graph, visual_groups, host_frame_id=frame_id, insert_values=False)

        # 对于窗口中已经放入边缘化图的路标点
        # 当前窗口中所有将边缘化帧作为host帧的路标点若没有被其他帧观测到，则认为该路标点死亡，反之则存活
        surviving = []
        dead = []
        for lm_id in hosted:
            seen = set(observation_frames.get(lm_id, ())) if observation_frames else {
                obs_frame for obs_frame, _ in visual_groups[lm_id]['observations']
            }
            (surviving if any(obs != frame_id for obs in seen) else dead).append(lm_id)

        # 对于窗口中没有放入边缘化图的路标点
        # 如果该路标点已经被边缘化帧观测到但没有被其他帧观测到，则认为该路标点死亡，反之则存活
        if observation_frames:
            for lm_id, seen_frames in observation_frames.items():
                if frame_id not in seen_frames or any(other != frame_id for other in seen_frames):
                    continue
                if lm_id not in dead:
                    dead.append(lm_id)

        # 收集边缘化图中的和边缘化帧相关的因子（边缘化帧自己的X/V/B和死亡路标）
        graph_keys = self._graph_keys(marg_graph)
        frame_keys = [X(gtsam_id), V(gtsam_id), B(gtsam_id)]
        if any(key not in graph_keys for key in frame_keys):
            print(f"【Backend】: Frame {frame_id} is missing pose, velocity, or bias factors. Skip marginalization.")
            return None

        # 旧 host 上的逆深度随该帧一起消除。
        keys_to_marg_list = []
        removed_landmarks = []
        for lm_id in surviving + dead:
            depth_key = self._inverse_depth_key(self._get_lm_gtsam_id(lm_id))
            # host帧相关的逆深度因子需要被边缘化
            if depth_key in graph_keys:
                keys_to_marg_list.append(depth_key)
                # 如果该路标点死亡，则需要被移除
                if lm_id in dead:
                    removed_landmarks.append(lm_id)

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

        # 挑出保留的边缘化因子键(不包含死亡路标点)
        if self.marg_factor is not None:
            kept_graph = gtsam.NonlinearFactorGraph()
            kept_graph.push_back(self.marg_factor)
            kept_keys = self._graph_keys(kept_graph)
        else:
            kept_keys = set()

        # 移除死亡路标点的逆深度因子以及该路标点在active_values中的状态
        for lm_id in dead:
            if lm_id in removed_landmarks:
                continue
            depth_key = self._inverse_depth_key(self._get_lm_gtsam_id(lm_id))
            if depth_key not in kept_keys and self.active_values.exists(depth_key):
                self.active_values.erase(depth_key)
                removed_landmarks.append(lm_id)
        self.stationary_frame_gtsam_ids.discard(gtsam_id)

        print(
            f"【Backend】: Marginalized oldest frame {frame_id} X({gtsam_id}), "
            f"kept {len(surviving)} landmarks, removed {len(removed_landmarks)} landmarks."
        )
        return {
            'success': True,
            'mode': 'MARGIN_OLD',
            'marginalized_frame_id': frame_id,
            # 只含死亡路标，供 LocalMap 删除。存活点已消除旧逆深度，仍留在地图中。
            'removed_landmark_ids': removed_landmarks,
            'kept_landmark_count': len(surviving),
            'consumed_imu_factor_ids': consumed_imu,
        }

    def marginalize_second_newest(self, frame_id, hosted_landmark_ids=None):
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
        hosted_depth_keys = [
            self._inverse_depth_key(self._get_lm_gtsam_id(lm_id))
            for lm_id in (hosted_landmark_ids or [])
        ]
        keys_to_marg_list = [
            key for key in frame_keys + hosted_depth_keys if key in prior_keys
        ]

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
        for key in hosted_depth_keys:
            if self.active_values.exists(key):
                self.active_values.erase(key)
        self.stationary_frame_gtsam_ids.discard(gtsam_id)

        print(f"【Backend】: Marginalized second-newest frame {frame_id} X({gtsam_id}).")
        return {
            'success': True,
            'mode': 'MARGIN_SECOND_NEW',
            'marginalized_frame_id': frame_id,
            'marginalized_gtsam_id': gtsam_id,
            # 次新帧不在这里列出要删除的路标，LocalMap 由 remove_frame 清理空观测点。
            'removed_landmark_ids': [],
        }

    def initialize_optimize(self, initial_keyframes, initial_imu_factors, initial_landmarks, initial_velocities, initial_bias):
        print("【Backend】: Initializing optimize...")

        # 确保初始化时状态机是干净的
        self.active_values.clear()
        self.active_frame_gtsam_ids.clear()
        self.optimized_landmark_positions.clear()
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

            # Fix only the pose gauge. VINS-Mono does not pin the initialized
            # velocity or accelerometer bias to their first estimates; doing
            # so here would permanently bake them into the first marginal
            # prior. Optional loose priors remain available for datasets that
            # explicitly need them.
            if frame_gtsam_id == 0:
                prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*3 + [1e-2]*3))
                f_pose = gtsam.PriorFactorPose3(X(0), T_wb, prior_pose_noise)
                self.init_priors.add(f_pose)

                velocity_sigma = self.config.get('initial_velocity_prior_sigma')
                if velocity_sigma is not None:
                    self.init_priors.add(gtsam.PriorFactorVector(
                        V(0), velocity,
                        gtsam.noiseModel.Isotropic.Sigma(3, float(velocity_sigma))))

                accel_bias_sigma = self.config.get('initial_accel_bias_prior_sigma')
                gyro_bias_sigma = self.config.get('initial_gyro_bias_prior_sigma')
                if accel_bias_sigma is not None or gyro_bias_sigma is not None:
                    accel_sigma = float(accel_bias_sigma if accel_bias_sigma is not None else 1e6)
                    gyro_sigma = float(gyro_bias_sigma if gyro_bias_sigma is not None else 1e6)
                    self.init_priors.add(gtsam.PriorFactorConstantBias(
                        B(0), bias, gtsam.noiseModel.Diagonal.Sigmas(
                            np.array([accel_sigma] * 3 + [gyro_sigma] * 3))))

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
            self._cache_optimized_inverse_depth_positions(visual_groups)
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

        if is_stationary and self.use_zupt:
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
            f"【Backend】: Frame {new_frame.get_id()} window graph has {current_graph.size()} factors "
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
            self._cache_optimized_inverse_depth_positions(visual_groups)
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
