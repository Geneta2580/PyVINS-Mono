import gtsam
import numpy as np
import threading
import queue
import time

from gtsam.symbol_shorthand import X, V, B

from .backend import Backend
from datatype.frame import Frame
from datatype.global_map import GlobalMap
from datatype.localmap import LocalMap
from datatype.landmark import Landmark, LandmarkStatus
from .imu_process import IMUProcessor
from .sfm_processor import SfMProcessor
from .viewer import Viewer3D
from utils.debug import Debugger
from .vio_initializer import VIOInitializer


class Estimator(threading.Thread):
    """
    The central coordinator for the SLAM system. Runs as a consumer thread.
    """
    # 与 VINS-Mono 一致：关键帧边缘化最老帧，普通帧边缘化次新帧
    MARGIN_OLD = 0
    MARGIN_SECOND_NEW = 1
    def __init__(self, config, imu_processor, input_queue, viewer_queue, global_central_map):
        super().__init__(daemon=True)
        self.config = config
        self.input_queue = input_queue
        self.local_map = LocalMap(config)

        # IMU相关
        self.imu_processor = imu_processor
        self.imu_buffer = []
        self.received_imu_count = 0
        self.cached_imu_edges = {}  # {(id1, id2): {pim, measurements, timestamps, bias_hat}}
        self.scheduled_frames = []  # 等到 IMU 覆盖该帧时间后再进入初始化或跟踪

        self.backend = Backend(global_central_map, config, self.imu_processor)

        # 读取相机内参
        cam_intrinsics_raw = self.config.get('cam_intrinsics', np.eye(3).flatten().tolist())
        self.cam_intrinsics = np.asarray(cam_intrinsics_raw).reshape(3, 3)

        self.sfm_processor = SfMProcessor(self.config, self.cam_intrinsics)

        self.next_f_id = 0

        # 初始化相关设置
        self.is_initialized = False

        self.init_window_size = self.config.get('init_window_size', 10)
        self.initial_parallax = self.config.get('initial_parallax', 40)
        self.init_feature_tracks = {}

        self.gravity_magnitude = self.config.get('gravity', 9.81)
        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)

        # 可视化test
        self.viewer_queue = viewer_queue

        # 是否使用IMU输出
        self.use_imu_output = self.config.get('use_imu_output', False)

        # 轨迹文件
        self.trajectory_file = None
        trajectory_output_path = self.config.get('trajectory_output_path', None) # self.config.get('trajectory_output_path', None)
        if trajectory_output_path:
            self.trajectory_file = Debugger.initialize_trajectory_file(trajectory_output_path)

        # 劣质点黑名单
        self.landmark_denylist = set()

        # 最后一个已处理帧的 id
        self.last_processed_f_id = -1

        # 最后一个处理的关键帧时间戳
        self.last_processed_normal_frame_id = -1

        # 最后一个处理IMU的时间戳
        self.last_processed_imu_timestamp = -1

        # 最新的导航状态（用于快速积分）
        self.latest_nav_state = None

        # Threading control
        self.is_running = False

    def start(self):
        self.is_running = True
        super().start()

    def shutdown(self):
        self.is_running = False
        if self.trajectory_file:
            self.trajectory_file.close()
            print("【Estimator】Trajectory file closed.")
        print("【Estimator】shut down.")

    def run(self):
        print("【Estimator】thread started.")
        while self.is_running:
            try:
                package = self.input_queue.get(timeout=1.0)

                if package is None:
                    print("【Estimator】received shutdown signal from frontend.")
                    print(f"【IMU Queue】received {self.received_imu_count}")
                    break

                imu_batch = package.get('imu_since_last_image')
                if imu_batch:
                    self._ingest_imu_batch(imu_batch)

                # 接收视觉特征点数据
                if 'visual_features' in package:
                    timestamp = package['timestamp']
                    visual_features = package['visual_features']
                    feature_ids = package['feature_ids']
                    image = package['image']
                    is_stationary = package['is_stationary']
                    is_keyframe = bool(package['is_kf'])
                    if is_stationary:
                        print(f"【Estimator】: Stationary frame detected at timestamp: {timestamp}")

                    # 初始化窗口只收关键帧，避免普通帧占满 window_size
                    if not self.is_initialized and not is_keyframe:
                        continue

                    # 关键帧和普通帧使用同一套帧表示
                    filtered_features = []
                    filtered_ids = []
                    for feat, fid in zip(visual_features, feature_ids):
                        if fid not in self.landmark_denylist:
                            filtered_features.append(feat)
                            filtered_ids.append(fid)

                    if len(filtered_ids) < 10: # (可选的安全检查)
                        print(f"【Estimator】: 过滤后特征点过少 ({len(filtered_ids)})，跳过此帧。")
                        continue

                    new_id = self.next_f_id
                    new_frame = Frame(new_id, timestamp)
                    new_frame.add_visual_features(filtered_features, filtered_ids)
                    new_frame.set_image(image)
                    new_frame.set_is_stationary(is_stationary)
                    new_frame.set_is_keyframe(is_keyframe)

                    self.next_f_id += 1
                    self._schedule_frame(new_frame, is_stationary, is_keyframe)

            except queue.Empty:
                continue
        
        print("【Estimator】thread has finished.")

    def _ingest_imu_batch(self, imu_batch):
        """图像包里的 IMU 按时间顺序写入。丢样本会在结束时和发送计数对不上。"""
        for timestamp, measurement in imu_batch:
            self.imu_buffer.append({
                'imu_measurements': measurement,
                'timestamp': timestamp,
            })
            self.received_imu_count += 1
            if self.use_imu_output and len(self.imu_buffer) >= 2:
                if timestamp > self.last_processed_imu_timestamp:
                    self.process_imu_data(timestamp)
                    self.last_processed_imu_timestamp = timestamp
        self._drain_scheduled_frames()

    def _schedule_frame(self, frame, is_stationary, is_keyframe):
        """图像时刻若还没有 IMU，就先挂起。EuRoC 常有一条与图像时间相同的 IMU，它可能排在图像之后。"""
        self.scheduled_frames.append((frame, is_stationary, is_keyframe))
        self._drain_scheduled_frames()

    def _imu_covers_timestamp(self, timestamp):
        """最新一条 IMU 是否已经不早于（晚于）该图像时刻。中值积分需要用它做右端点。"""
        if not self.imu_buffer:
            return False
        return self.imu_buffer[-1]['timestamp'] + 1e-9 >= timestamp

    def _drain_scheduled_frames(self):
        """从队列中取出最早的帧，并将其添加到LocalMap中。"""
        while self.scheduled_frames:
            frame, is_stationary, is_keyframe = self.scheduled_frames[0]
            if not self._imu_covers_timestamp(frame.get_timestamp()):
                return
            self.scheduled_frames.pop(0)
            self.local_map.add_frame(frame)
            # 如果未初始化，则收集帧，直到达到初始化窗口大小
            if not self.is_initialized:
                active_frames = self.local_map.get_active_frames()
                if len(active_frames) == self.init_window_size:
                    self.visual_inertial_initialization()
                else:
                    print(f"【Init】: Collecting frames... {len(active_frames)}/{self.init_window_size}")
            # 如果已初始化，则处理帧
            else:
                self.process_frame_data(frame, is_stationary, is_keyframe)
                self.last_processed_f_id = frame.get_id()

    def marginalization_flag(self, is_keyframe):
        if is_keyframe:
            return self.MARGIN_OLD
        return self.MARGIN_SECOND_NEW

    def _drop_imu_cache_for_frame(self, frame_id):
        """删掉连接到该帧的预积分。MARGIN_OLD 中这些因子已经进入先验，不再补跨帧预积分。"""
        stale_keys = [key for key in self.cached_imu_edges if frame_id in key]
        for key in stale_keys:
            del self.cached_imu_edges[key]

    def _store_imu_edge(self, frame_id_1, frame_id_2, pim, measurements, start_timestamp, end_timestamp, bias_hat):
        self.cached_imu_edges[(frame_id_1, frame_id_2)] = {
            'pim': pim,
            'measurements': measurements,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'bias_hat': bias_hat,
        }

    def _merge_imu_measurements(self, measurements_ab, measurements_bc):
        """按时间拼接两段原始 IMU，同一时间戳只保留一次。"""
        merged = {}
        for timestamp, measurement in list(measurements_ab) + list(measurements_bc):
            merged[timestamp] = measurement
        return [(timestamp, merged[timestamp]) for timestamp in sorted(merged)]

    def _build_bridging_imu_edge(self, frame_a, frame_b, frame_c, bias_a):
        edge_ab = self.cached_imu_edges.get((frame_a.get_id(), frame_b.get_id()))
        edge_bc = self.cached_imu_edges.get((frame_b.get_id(), frame_c.get_id()))
        if edge_ab is None or edge_bc is None:
            print(
                f"【Estimator】: Missing IMU edge for "
                f"{frame_a.get_id()}->{frame_b.get_id()} or {frame_b.get_id()}->{frame_c.get_id()}."
            )
            return None
        start_ts = frame_a.get_timestamp()
        end_ts = frame_c.get_timestamp()
        measurements = self._bounded_imu_between(start_ts, end_ts)
        if measurements is None:
            measurements = self._merge_imu_measurements(edge_ab['measurements'], edge_bc['measurements'])
        pim = self.imu_processor.pre_integration(
            measurements, start_ts, end_ts, override_bias=bias_a)
        if pim is None:
            return None
        return {
            'pim': pim,
            'measurements': measurements,
            'start_timestamp': frame_a.get_timestamp(),
            'end_timestamp': frame_c.get_timestamp(),
            'bias_hat': bias_a,
        }

    def _drop_marginalized_frame(self, frame_id):
        # 删除连接到该帧的IMU预积分
        self._drop_imu_cache_for_frame(frame_id)
        # 删除LocalMap中的帧
        self.local_map.remove_frame(frame_id)

    def _apply_margin_result(self, result):
        """边缘化成功后才删除同一帧，以及已经随该帧消掉的 host 路标。"""
        if not isinstance(result, dict) or not result.get('success'):
            return
        frame_id = result.get('marginalized_frame_id')
        if frame_id is None:
            return
        landmark_ids = result.get('marginalized_landmark_ids', [])
        print(
            f"【Estimator】: Commit marginalization of frame {frame_id} "
            f"and {len(landmark_ids)} host landmarks."
        )
        self._drop_marginalized_frame(frame_id)
        for lm_id in landmark_ids:
            self.local_map.landmarks.pop(lm_id, None)
        self._assert_window_consistency()

    def _commit_margin_second_new(self, frame_a, frame_b, frame_c, edge_ac):
        id_a = frame_a.get_id()
        id_b = frame_b.get_id()
        id_c = frame_c.get_id()
        del self.cached_imu_edges[(id_a, id_b)]
        del self.cached_imu_edges[(id_b, id_c)]
        self.cached_imu_edges[(id_a, id_c)] = edge_ac
        self.local_map.remove_frame(id_b, transfer_host=True)
        print(f"【Estimator】: merged IMU {id_a} -> {id_b} -> {id_c} into {id_a} -> {id_c}")
        print(f"【Estimator】: marginalized second-newest frame {id_b}")
        self._assert_window_consistency()
        self._assert_second_new_removed(frame_a, frame_b, frame_c)

    def _assert_window_consistency(self):
        # 检查后端和LocalMap中的GTSAM帧ID是否一致
        backend_ids = [
            self.backend._frame_id_for_gtsam_id(gtsam_id)
            for gtsam_id in self.backend.active_frame_gtsam_ids
        ]
        local_ids = [frame.get_id() for frame in self.local_map.get_active_frames()]
        assert backend_ids == local_ids, (
            f"Backend frames {backend_ids} differ from LocalMap frames {local_ids}"
        )
        for left_id, right_id in zip(local_ids, local_ids[1:]):
            assert (left_id, right_id) in self.cached_imu_edges, (
                f"Missing IMU edge {left_id} -> {right_id}"
            )

    def _assert_second_new_removed(self, frame_a, frame_b, frame_c):
        id_b = frame_b.get_id()
        assert id_b not in self.local_map.frames
        assert (frame_a.get_id(), id_b) not in self.cached_imu_edges
        assert (id_b, frame_c.get_id()) not in self.cached_imu_edges
        assert (frame_a.get_id(), frame_c.get_id()) in self.cached_imu_edges
        for landmark in self.local_map.landmarks.values():
            assert id_b not in landmark.observations
            assert landmark.host_frame_id != id_b
        gtsam_id = self.backend.frame_id_to_gtsam_id.get(id_b)
        if gtsam_id is None or self.backend.marg_factor is None:
            return
        prior_graph = gtsam.NonlinearFactorGraph()
        prior_graph.push_back(self.backend.marg_factor)
        prior_keys = self.backend._graph_keys(prior_graph)
        assert X(gtsam_id) not in prior_keys
        assert V(gtsam_id) not in prior_keys
        assert B(gtsam_id) not in prior_keys

    def _discard_oldest_frame(self):
        active_frames = self.local_map.get_active_frames()
        if not active_frames:
            return
        oldest_frame = min(active_frames, key=lambda frame: frame.get_timestamp())
        self._drop_marginalized_frame(oldest_frame.get_id())

    def _bounded_imu_between(self, start_ts, end_ts):
        timed_measurements = [
            (pkg['timestamp'], pkg['imu_measurements']) for pkg in self.imu_buffer
        ]
        return self.imu_processor.build_bounded_imu_samples(timed_measurements, start_ts, end_ts)

    def create_imu_factors(self, frame_start, frame_end, log_stats=False):
        start_ts = frame_start.get_timestamp()
        end_ts = frame_end.get_timestamp()
        measurements_with_ts = self._bounded_imu_between(start_ts, end_ts)

        if not measurements_with_ts:
            print(f"【Estimator】: No IMU measurements between frame {frame_start.get_id()} and frame {frame_end.get_id()}.")
            return None

        bias_hat = self.imu_processor.current_bias
        imu_preintegration = self.imu_processor.pre_integration(
            measurements_with_ts, start_ts, end_ts, log_stats=log_stats)

        if imu_preintegration:
            return {
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'imu_measurements': measurements_with_ts,
                'imu_preintegration': imu_preintegration,
                'bias_hat': bias_hat,
            }

        return None
        
    # TODO:def check_motion_excitement(self):

    def triangulate_new_landmarks(self):
        newly_triangulated_for_backend = {}
        frame_window = self.local_map.get_active_frames()
        for lm in self.local_map.get_candidate_landmarks():            

            is_ready, first_frame, last_frame = lm.is_ready_for_triangulation(frame_window, min_parallax=40)
            
            if is_ready:
                T_w_b_1 = first_frame.get_global_pose()
                T_w_b_2 = last_frame.get_global_pose()
                
                if T_w_b_1 is None or T_w_b_2 is None:
                    continue

                T_w_c_1 = T_w_b_1 @ self.T_bc
                T_w_c_2 = T_w_b_2 @ self.T_bc
                T_c_2_1 = np.linalg.inv(T_w_c_2) @ T_w_c_1

                R, t = T_c_2_1[:3, :3], T_c_2_1[:3, 3].reshape(3, 1)

                pts1 = np.array([lm.get_observation(first_frame.get_id())])
                pts2 = np.array([lm.get_observation(last_frame.get_id())])

                points_3d_in_c1, mask = self.sfm_processor.triangulate_points(pts1, pts2, R, t)

                if len(points_3d_in_c1) > 0:
                    points_3d_world = (T_w_c_1[:3, :3] @ points_3d_in_c1.T + T_w_c_1[:3, 3].reshape(3, 1)).flatten()
                    is_healthy = self.local_map.check_landmark_health(lm.id, points_3d_world)
                    if is_healthy:
                        lm.set_triangulated(points_3d_world)
                        newly_triangulated_for_backend[lm.id] = points_3d_world
                    
                    else:
                        continue
                
                else:
                    continue
    
        return newly_triangulated_for_backend
            
    
    # 审计地图，移除所有变得不健康的坏点
    def audit_map_after_optimization(self, oldest_frame_id_in_window):
        landmarks_to_remove = []
        landmarks_to_remove_depth = []
        landmarks_to_remove_reproj = []
        # 遍历所有已三角化的路标点
        for lm_id in self.local_map.get_active_landmarks().keys():
            is_health_ok, is_depth_ok, is_reproj_ok = self.local_map.check_landmark_health_after_optimization(lm_id)
            if not is_health_ok:
                landmarks_to_remove.append(lm_id)
            if not is_depth_ok:
                landmarks_to_remove_depth.append(lm_id)
            if not is_reproj_ok:
                landmarks_to_remove_reproj.append(lm_id)

        if landmarks_to_remove:
            print(f"【Audit】: Removing {len(landmarks_to_remove)} landmarks that became unhealthy after optimization: {landmarks_to_remove}")
            # 从LocalMap中删除
            for lm_id in landmarks_to_remove:
                if lm_id in self.local_map.landmarks:
                    del self.local_map.landmarks[lm_id]

                # 将其列入黑名单
                self.landmark_denylist.add(lm_id)

            # 从后端移除异常点
            self.backend.remove_stale_landmarks(landmarks_to_remove, landmarks_to_remove_depth, 
                                                landmarks_to_remove_reproj, oldest_frame_id_in_window)
    
    # 零速检查IMU
    def is_stationary(self, imu_measurements_between_frames):
        if len(imu_measurements_between_frames) < 10: # 至少需要一些样本
            return False

        # 提取所有的加速度和角速度读数
        accel_list = [m[1].accel.astype(np.float64) for m in imu_measurements_between_frames]
        gyro_list = [m[1].gyro.astype(np.float64) for m in imu_measurements_between_frames]

        try:
            accels = np.array(accel_list, dtype=np.float64)         
            gyros = np.array(gyro_list, dtype=np.float64)
        except ValueError:
            print("【Stationary Check】: Failed to convert IMU measurements to numpy arrays.")
            return False

        # 计算加速度和角速度在每个轴上的标准差
        accel_std = np.std(accels, axis=0)
        gyro_std = np.std(gyros, axis=0)

        # 从config中读取阈值
        accel_std_threshold = self.config.get('stationary_accel_std_threshold', 0.05) # m/s^2
        gyro_std_threshold = self.config.get('stationary_gyro_std_threshold', 0.05) # rad/s

        # 如果所有轴的波动都小于阈值，则认为是静止
        is_still = np.all(accel_std < accel_std_threshold) and np.all(gyro_std < gyro_std_threshold)

        if is_still:
            print("【Stationary Check】: System is stationary.")
            
        return is_still

    # 收集当前滑窗内所有相邻帧之间的 IMU 预积分因子用于优化
    def _gather_window_imu_factors(self, active_frames):
        window_imu_factors = []
        for i in range(len(active_frames) - 1):
            id1 = active_frames[i].get_id()
            id2 = active_frames[i+1].get_id()
            
            # 从缓存中直接拿，时间复杂度 O(1)
            edge = self.cached_imu_edges.get((id1, id2))
            if edge is not None:
                window_imu_factors.append({
                    'frame_id1': id1,
                    'frame_id2': id2,
                    'pim': edge['pim']
                })
            else:
                print(f"【Warning】丢失了帧 {id1} 到帧 {id2} 的 IMU 预积分！")
        return window_imu_factors

    # 收集当前滑窗内所有健康的 3D 路标点及它们的 2D 观测用于优化
    def _gather_window_visual_data(self, active_frames):
        active_frame_ids = {frame.get_id() for frame in active_frames}
        window_landmarks = {}       # {lm_id: 3d_position}
        window_visual_factors = []  # [(frame_id, lm_id, pt_2d), ...]

        for lm_id, lm in self.local_map.landmarks.items():
            # 1. 过滤黑名单
            if lm_id in self.landmark_denylist:
                continue
            
            # 2. 只打包已经成功三角化的点
            if lm.status != LandmarkStatus.TRIANGULATED:
                continue

            # 3. 找出这个点在当前【活跃窗口】内的所有观测
            obs_in_window = [(k_id, pt) for k_id, pt in lm.observations.items() if k_id in active_frame_ids]

            # 4. 如果这个点在当前窗口内可见，才把它打包进去
            if len(obs_in_window) > 0:
                window_landmarks[lm_id] = lm.position_3d.copy() # 深拷贝阻断共享，前端有可能动lm
                for k_id, pt in obs_in_window:
                    pt_safe = pt.copy() if hasattr(pt, 'copy') else pt
                    window_visual_factors.append((k_id, lm_id, pt_safe))

        return window_landmarks, window_visual_factors


    def visual_inertial_initialization(self):
        print("【Init】: Buffer is full. Starting initialization process.")

        initial_keyframes = self.local_map.get_active_frames()
        # 视觉初始化（SFM）
        sfm_success = self.visual_initialization(initial_keyframes)

        # 视觉初始化失败，滑动窗口继续初始化
        if not sfm_success:
            print("【Init】: Visual initialization failed. Sliding window.")
            self._discard_oldest_frame()
            return

        # 创建初始化IMU因子
        initial_imu_factors = []
        for i in range(len(initial_keyframes) - 1):
            kf_start = initial_keyframes[i]
            kf_end = initial_keyframes[i + 1]
            imu_factors = self.create_imu_factors(kf_start, kf_end, log_stats=True)
            if imu_factors is None:
                print("【Init】: IMU interval is incomplete. Sliding window.")
                self._discard_oldest_frame()
                return
            if imu_factors:
                initial_imu_factors.append(imu_factors)
                self._store_imu_edge(
                    kf_start.get_id(), kf_end.get_id(),
                    imu_factors['imu_preintegration'],
                    imu_factors['imu_measurements'],
                    imu_factors['start_timestamp'],
                    imu_factors['end_timestamp'],
                    imu_factors['bias_hat'],
                )

        # 视觉惯性初始化
        alignment_success, scale, gyro_bias, velocities, gravity_w = VIOInitializer.initialize(
            initial_keyframes, 
            initial_imu_factors, 
            self.imu_processor, 
            self.gravity_magnitude, 
            self.T_bc
        )

        if alignment_success:
            # 尺度对齐之后，用公制相机位姿重三角化全部初始化轨迹。
            self._retriangulate_metric_tracks(initial_keyframes)
            print("【Init】: Alignment successful. Calling backend to build initial graph...")

            # 更新IMU偏置
            initial_bias_obj = gtsam.imuBias.ConstantBias(np.zeros(3), gyro_bias)
            self.imu_processor.update_bias(initial_bias_obj)
            
            # test
            poses = {kf.get_id(): kf.get_global_pose() for kf in initial_keyframes if kf.get_global_pose() is not None}
            for kf_id, pose in poses.items():
                print(f"【Init】: Before optimization. kf_id: {kf_id}, pose: {pose[:3, 3]}")
            # test

            # 进行初始优化。成功后才滑掉最老帧和它的 host 路标。
            margin_result = self.backend.initialize_optimize(
                self.local_map.get_active_frames(),
                initial_imu_factors,
                self.local_map.get_active_landmarks(),
                velocities, initial_bias_obj
            )

            # 初始优化结束，同步后端结果到Estimator
            self.backend.update_estimator_map(
                self.local_map.get_active_frames(),
                self.local_map.landmarks
            )
            self._apply_margin_result(margin_result) # 同步边缘化结果到LocalMap
            self._assert_window_consistency()
            self.is_initialized = True
            self.last_processed_f_id = initial_keyframes[-1].get_id()

            # 用后端优化结果初始化最新的导航状态
            latest_pose, latest_velocity, _ = self.backend.get_latest_optimized_state()
            if latest_pose is not None and latest_velocity is not None:
                latest_vel_np = np.array(latest_velocity) if not isinstance(latest_velocity, np.ndarray) else latest_velocity
                self.latest_nav_state = {
                    'pose': latest_pose,
                    'velocity': latest_vel_np
                }
                print(f"【Init】: Initialized latest_nav_state from backend optimization")

            # 记录初始优化轨迹
            if self.trajectory_file:
                print("【Estimator】正在记录初始优化轨迹...")
                # 按时间戳排序以确保轨迹顺序正确
                sorted_frames = sorted(self.local_map.get_active_frames(), key=lambda frame: frame.get_timestamp())
                for frame in sorted_frames:
                    Debugger.log_trajectory_tum(self.trajectory_file, frame) # 调用静态方法
                self.trajectory_file.flush() # 确保数据立即写入磁盘
            # 记录初始优化轨迹
            
            #  viewer可视化
            if self.viewer_queue:
                print("【Init】: Sending initialization result to viewer queue...")

                # 从 local_map 中获取最新的、优化后的位姿和路标点数据
                active_frames = self.local_map.get_active_frames()
                poses = {frame.get_id(): frame.get_global_pose() for frame in active_frames if frame.get_global_pose() is not None}
                
                # 调用 LocalMap 的辅助函数来获取纯粹的位置字典
                landmarks_positions = self.local_map.get_active_landmarks()

                vis_data = {
                    'landmarks': landmarks_positions,
                    'poses': poses
                }
                
                # 打印一些信息以供调试
                print(f"【Viewer】: Sending {len(poses)} poses and {len(landmarks_positions)} landmarks to viewer.")

                try:
                    self.viewer_queue.put_nowait(vis_data)
                except queue.Full:
                    print("【Estimator】: Viewer queue is full, skipping visualization data.")
            # viewer可视化
            
        else:
            print("【Init】: V-I Alignment failed.")
            self._discard_oldest_frame()

        return alignment_success
    
    def _retriangulate_metric_tracks(self, frames):
        """尺度对齐之后，用公制相机位姿重三角化全部初始化轨迹。"""
        kept = 0
        removed = 0
        for feature_id, observations in list(self.init_feature_tracks.items()):
            pose_list = [None] * len(frames)
            posed_observations = []
            for frame_idx, uv in observations:
                if frame_idx >= len(frames):
                    continue
                body_pose = frames[frame_idx].get_global_pose()
                if body_pose is None:
                    continue
                pose_list[frame_idx] = body_pose @ self.T_bc
                posed_observations.append((frame_idx, uv))
            landmark = self.local_map.landmarks.get(feature_id)
            point = None
            if landmark is not None and len(posed_observations) >= 2:
                point = self.sfm_processor.triangulate_track(posed_observations, pose_list, max_error=5.0)
            if point is None:
                self.local_map.landmarks.pop(feature_id, None)
                removed += 1
                continue
            landmark.set_triangulated(point)
            kept += 1
        print(
            f"【Init】: Re-triangulated {kept} tracks, removed {removed}. "
            f"Final map has {len(self.local_map.landmarks)} landmarks."
        )

    def visual_initialization(self, initial_keyframes):
        print("【Visual Init】: Global SFM on the initialization window...")
        result = self.sfm_processor.initialize_window(initial_keyframes, self.initial_parallax)
        if result is None:
            self.init_feature_tracks = {}
            return False

        self.init_feature_tracks = result['tracks']
        self.local_map.landmarks.clear()
        for feature_id, observations in result['tracks'].items():
            if feature_id not in result['landmarks']:
                continue
            first_idx, first_uv = observations[0]
            host = initial_keyframes[first_idx]
            landmark = Landmark(feature_id, host.get_id(), first_uv)
            for frame_idx, uv in observations[1:]:
                landmark.add_observation(initial_keyframes[frame_idx].get_id(), uv)
            landmark.set_triangulated(result['landmarks'][feature_id])
            self.local_map.landmarks[feature_id] = landmark
        print(f"【Visual Init】: Success! Map has {len(self.local_map.landmarks)} landmarks.")
        return True

    def process_frame_data(self, new_frame, is_stationary, is_keyframe):
        margin_flag = self.marginalization_flag(is_keyframe)
        margin_name = "MARGIN_OLD" if margin_flag == self.MARGIN_OLD else "MARGIN_SECOND_NEW"
        print(
            f"【Estimator】: Frame {new_frame.get_id()} "
            f"is_keyframe={is_keyframe} margin={margin_name}"
        )

        active_frames = self.local_map.get_active_frames()
        if len(active_frames) < 2:
            return

        last_frame = active_frames[-2]

        # ---------------------------------------------------------
        # 1. 计算最新的一段 IMU 预积分并存入缓存（创建上一帧到当前帧的IMU因子）
        # ---------------------------------------------------------
        start_time = time.time()
        imu_factor_data = self.create_imu_factors(last_frame, new_frame)
        end_time = time.time()
        print(f"【Estimator Timer】: IMU Factor Creation took {(end_time - start_time) * 1000:.3f} ms.")
        if imu_factor_data:
            self._store_imu_edge(
                last_frame.get_id(), new_frame.get_id(),
                imu_factor_data['imu_preintegration'],
                imu_factor_data['imu_measurements'],
                imu_factor_data['start_timestamp'],
                imu_factor_data['end_timestamp'],
                imu_factor_data['bias_hat'],
            )
        else:
            print(f"【Estimator】: No IMU factors between frame {last_frame.get_id()} and frame {new_frame.get_id()}.")
            self._drop_marginalized_frame(new_frame.get_id())
            return

        # is_currently_stationary = self.is_stationary(imu_factor_data['imu_measurements']) IMU零速检查
        is_currently_stationary = is_stationary

        # ---------------------------------------------------------
        # 2. 状态预测与新点三角化
        # ---------------------------------------------------------
        # 从后端获取最新的优化结果
        last_pose, last_vel, last_bias = self.backend.get_latest_optimized_state()
        if last_pose is None:
            print(f"[Warning] Could not retrieve last state from backend. Skipping frame {new_frame.get_id()}.")
            self._drop_marginalized_frame(new_frame.get_id())
            return

        # 使用当前帧的预积分来预测当前帧状态
        pim = imu_factor_data['imu_preintegration']
        predicted_nav_state = pim.predict(gtsam.NavState(last_pose, last_vel), last_bias)

        predicted_T_wb = predicted_nav_state.pose()
        predicted_vel = predicted_nav_state.velocity()

        # 预测位姿设置当前帧初值
        new_frame.set_global_pose(predicted_T_wb.matrix())

        # 进行新特征点三角化
        new_landmarks = self.triangulate_new_landmarks()
        if new_landmarks:
            print(f"【Tracking】: Triangulated {len(new_landmarks)} new landmarks.")
            print(f"【Tracking】: New landmarks: {new_landmarks.keys()}")
        
        # ---------------------------------------------------------
        # 3. 生成 Backend 优化滑窗所需的数据包
        # ---------------------------------------------------------
        # 收集所有的 IMU 预积分
        window_imu_factors = self._gather_window_imu_factors(active_frames)

        # 收集所有健康的 3D 点和 2D 观测
        window_landmarks, window_visual_factors = self._gather_window_visual_data(active_frames)
        # 收集所有路标点的 host frame ID（用于边缘化）
        landmark_hosts = {
            lm_id: self.local_map.landmarks[lm_id].host_frame_id
            for lm_id in window_landmarks
            if lm_id in self.local_map.landmarks
        }

        # ---------------------------------------------------------
        # 4. 完整窗口优化。边缘化类型由当前帧是否为关键帧决定。
        # ---------------------------------------------------------
        start_time = time.time()
        opt_result = self.backend.optimize_window(
            active_frames=active_frames,
            window_imu_factors=window_imu_factors,
            window_landmarks=window_landmarks,
            window_visual_factors=window_visual_factors,
            new_frame_initial_guess=(predicted_T_wb, predicted_vel, last_bias),
            is_stationary=is_currently_stationary,
            landmark_hosts=landmark_hosts
        )
        end_time = time.time()
        print(f"【Estimator Timer】: Backend Incremental Optimization took {(end_time - start_time) * 1000:.3f} ms.")
        if not opt_result or not opt_result.get('success'):
            return

        # ---------------------------------------------------------
        # 5. 先同步优化结果，再按帧类型提交边缘化
        # ---------------------------------------------------------

        # 同步后端优化结果到LocalMap
        self.backend.update_estimator_map(active_frames, self.local_map.landmarks)

        # 边缘化最老帧
        if margin_flag == self.MARGIN_OLD:
            margin_result = self.backend.marginalize_oldest(
                window_imu_factors, opt_result['visual_groups'])
            if margin_result and margin_result.get('success'):
                self._apply_margin_result(margin_result)
            else:
                print("【Estimator】: Oldest-frame marginalization failed. Keep the window.")

        # 边缘化第二新帧
        else:
            if len(active_frames) < 3:
                print("【Estimator】: Not enough frames for MARGIN_SECOND_NEW. Keep the window.")
            else:
                frame_a = active_frames[-3]
                frame_b = active_frames[-2]
                frame_c = active_frames[-1]
                _, _, bias_a = self.backend.get_optimized_state(frame_a.get_id())
                edge_ac = None if bias_a is None else self._build_bridging_imu_edge(
                    frame_a, frame_b, frame_c, bias_a)
                if edge_ac is None:
                    print("【Estimator】: Failed to reintegrate PIM. Keep the second-newest frame.")
                else:
                    margin_result = self.backend.marginalize_second_newest(frame_b.get_id())
                    if margin_result and margin_result.get('success'):
                        self._commit_margin_second_new(frame_a, frame_b, frame_c, edge_ac)
                    else:
                        print("【Estimator】: Second-newest marginalization failed. Keep frame and IMU edges.")
        
        # 审计地图，移除所有变得不健康的"坏苹果"
        start_time = time.time()
        # self.audit_map_after_optimization(oldest_frame_id_in_window)
        end_time = time.time()
        print(f"【Estimator Timer】: Map Audit took {(end_time - start_time) * 1000:.3f} ms.")
        
        # 更新预积分器的零偏
        _, _, latest_bias = self.backend.get_latest_optimized_state()
        if latest_bias:
            self.imu_processor.update_bias(latest_bias)

        # ---------------------------------------------------------
        # 6. 记录优化结果以及可视化
        # ---------------------------------------------------------
        # 用后端优化结果更新最新的导航状态（位姿和速度）
        latest_pose, latest_velocity, _ = self.backend.get_latest_optimized_state()
        if latest_pose is not None and latest_velocity is not None:
            # 将速度转换为 numpy 数组
            latest_vel_np = np.array(latest_velocity) if not isinstance(latest_velocity, np.ndarray) else latest_velocity
            self.latest_nav_state = {
                'pose': latest_pose,
                'velocity': latest_vel_np
            }
            print(f"【Estimator】: Updated latest_nav_state from backend optimization")

        # 记录优化轨迹
        Debugger.log_trajectory_tum(self.trajectory_file, new_frame) # 调用静态方法
        if self.trajectory_file:
            self.trajectory_file.flush() # 确保数据立即写入磁盘
        # 记录优化轨迹
        
        # viewer可视化
        if self.viewer_queue:
            print("【Tracking】: Sending tracking result to viewer queue...")

            # 从 local_map 中获取最新的、优化后的位姿和路标点数据
            active_frames = self.local_map.get_active_frames()
            poses = {frame.get_id(): frame.get_global_pose() for frame in active_frames if frame.get_global_pose() is not None}
            
            # 【核心修正】调用 LocalMap 的辅助函数来获取纯粹的位置字典
            landmarks_positions = self.local_map.get_active_landmarks()

            vis_data = {
                'landmarks': landmarks_positions,
                'poses': poses
            }
            
            # 打印一些信息以供调试
            print(f"【Viewer】: Sending {len(poses)} poses and {len(landmarks_positions)} landmarks to viewer.")

            try:
                self.viewer_queue.put_nowait(vis_data)
            except queue.Full:
                print("【Estimator】: Viewer queue is full, skipping visualization data.")

    def process_imu_data(self, latest_imu_timestamp):
        # 检查 latest_nav_state 是否已初始化
        if self.latest_nav_state is None:
            # 尝试从后端获取最新状态
            latest_pose, latest_velocity, _ = self.backend.get_latest_optimized_state()
            if latest_pose is not None and latest_velocity is not None:
                latest_vel_np = np.array(latest_velocity) if not isinstance(latest_velocity, np.ndarray) else latest_velocity
                self.latest_nav_state = {
                    'pose': latest_pose,
                    'velocity': latest_vel_np
                }
            else:
                print("【Warning】: latest_nav_state not initialized and cannot get from backend")
                return
        
        latest_imu_data = self.imu_buffer[-1]
        last_imu_data = self.imu_buffer[-2]

        last_imu_timestamp = last_imu_data['timestamp']
        dt = latest_imu_timestamp - last_imu_timestamp

        if dt > 0:
            current_pose, current_velocity = self.imu_processor.fast_integration(
                dt, self.latest_nav_state, latest_imu_data['imu_measurements']
            )
            # 更新 latest_nav_state
            self.latest_nav_state = {
                'pose': current_pose,
                'velocity': current_velocity
            }
            # 记录快速积分的位姿到轨迹文件
            if self.trajectory_file:
                Debugger.log_pose_tum(self.trajectory_file, latest_imu_timestamp, current_pose)
                # 注意：这里不立即flush，让系统自动缓冲，提高性能
                # 如果需要实时写入，可以取消下面的注释
                # self.trajectory_file.flush()
            
            return