import gtsam
import cv2
import numpy as np
import threading
import queue
import time
from bisect import bisect_right
from collections import deque
from pathlib import Path

from gtsam.symbol_shorthand import X, V, B

from .backend import Backend
from datatype.frame import Frame
from datatype.localmap import LocalMap
from datatype.landmark import Landmark, LandmarkStatus
from .imu_process import IMUProcessor
from .sfm_processor import SfMProcessor
from utils.debug import Debugger
from .vio_initializer import VIOInitializer


class Estimator(threading.Thread):
    """
    The central coordinator for the SLAM system. Runs as a consumer thread.
    """
    # 与 VINS-Mono 一致：关键帧边缘化最老帧，普通帧边缘化次新帧
    MARGIN_OLD = 0
    MARGIN_SECOND_NEW = 1
    def __init__(self, config, imu_processor, input_queue, viewer_queue, global_central_map,
                 loop_detector=None, loop_result_queue=None, gui_dispatcher=None,
                 loop_optimization=None, loop_optimization_result_queue=None):
        super().__init__(daemon=True)
        self.config = config
        self.input_queue = input_queue
        self.global_map = global_central_map
        self.local_map = LocalMap(config)

        # IMU相关
        self.imu_processor = imu_processor
        self.imu_buffer = deque()
        self.received_imu_count = 0
        self.cached_imu_edges = {}  # {(id1, id2): {pim, measurements, timestamps, bias_hat}}
        self.scheduled_frames = []  # 等到 IMU 覆盖该帧时间后再进入初始化或跟踪

        self.backend = Backend(global_central_map, config, self.imu_processor)

        # 读取相机内参
        cam_intrinsics_raw = self.config.get('cam_intrinsics', np.eye(3).flatten().tolist())
        self.cam_intrinsics = np.asarray(cam_intrinsics_raw).reshape(3, 3)

        self.sfm_processor = SfMProcessor(self.config, self.cam_intrinsics)

        print(
            "【Visual Geometry】: "
            "frontend=pinhole initialization=pinhole "
            "triangulation=pinhole backend=bearing"
        )

        self.next_f_id = 0

        # 初始化相关设置
        self.is_initialized = False

        self.init_window_size = self.config.get('init_window_size', 10)
        self.initial_parallax = self.config.get('initial_parallax', 40)
        self.keyframe_parallax = float(self.config.get('keyframe_parallax', 10.0))
        self.keyframe_min_tracked = int(self.config.get('keyframe_min_tracked', 20))
        self.init_quality_gate = bool(self.config.get('init_quality_gate', True))
        self.init_min_alignment_attempts = int(self.config.get('init_min_alignment_attempts', 2))
        self.init_alignment_attempts = 0
        self.previous_init_scale = None
        self.init_feature_tracks = {}

        self.gravity_magnitude = self.config.get('gravity', 9.81)
        T_bc_raw = self.config.get('T_bc', np.eye(4).flatten().tolist())
        self.T_bc = np.asarray(T_bc_raw).reshape(4, 4)

        # 可视化test
        self.viewer_queue = viewer_queue
        self.loop_detector = loop_detector
        self.loop_result_queue = loop_result_queue
        self.loop_optimization = loop_optimization
        self.loop_optimization_result_queue = loop_optimization_result_queue
        self.gui_dispatcher = gui_dispatcher
        self.latest_loop_detection = None
        self.latest_loop_optimization = None
        self.loop_optimization_busy = False
        self.pending_loop_detection = None
        self.loop_optimization_task_id = 0
        self.loop_optimization_shutdown_wait = max(
            0.0, float(config.get('loop_optimization_shutdown_wait_sec', 2.0))
        )
        # Estimator内部继续使用连续VIO坐标；该变换只负责VIO到全局PGO坐标的衔接。
        self.loop_drift = np.eye(4, dtype=np.float64)
        self.loop_submission_warning_printed = False
        self.loop_visualize = bool(config.get('loop_detector_visualize', False))
        self.loop_visualization_window = 'Loop Detection: current | best match'

        # 是否使用IMU输出
        self.use_imu_output = self.config.get('use_imu_output', False)

        # 轨迹文件
        self.trajectory_file = None
        trajectory_output_path = self.config.get('trajectory_output_path', None) # self.config.get('trajectory_output_path', None)
        if trajectory_output_path:
            self.trajectory_file = Debugger.initialize_trajectory_file(trajectory_output_path)
        corrected_path = self.config.get(
            'loop_corrected_trajectory_output_path', None
        )
        if corrected_path is None and trajectory_output_path:
            online_path = Path(trajectory_output_path)
            suffix = online_path.suffix or '.txt'
            corrected_path = online_path.with_name(
                f"{online_path.stem}_loop_corrected{suffix}"
            )
        self.loop_corrected_trajectory_path = (
            Path(corrected_path) if corrected_path else None
        )
        if (self.loop_corrected_trajectory_path is not None
                and trajectory_output_path is not None
                and self.loop_corrected_trajectory_path.resolve()
                == Path(trajectory_output_path).resolve()):
            raise ValueError(
                "loop_corrected_trajectory_output_path must differ from "
                "trajectory_output_path"
            )
        # 全部相机帧的最终全局位姿。它独立于LocalMap/GlobalMap生命周期，
        # 因此被MARGIN_SECOND_NEW丢弃的非关键帧也能出现在最终轨迹中。
        self.loop_trajectory_records = {}
        self.loop_trajectory_exported = False

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
                # 消费回环检测结果（若有，非阻塞）
                self._drain_loop_optimization_results()
                self._drain_loop_results()
                package = self.input_queue.get(timeout=1.0)

                if package is None:
                    self._drain_loop_optimization_results()
                    self._drain_loop_results()
                    self._finish_loop_optimization() # 系统结束运行时，等待回环优化结果，超时直接退出
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
                    if is_stationary:
                        print(f"【Estimator】: Stationary frame detected at timestamp: {timestamp}")

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
                    tracked_ids = np.asarray(
                        package.get('tracked_feature_ids', []), dtype=np.int64
                    ).reshape(-1)
                    tracked_raw = np.asarray(
                        package.get('tracked_raw_features', []), dtype=np.float32
                    ).reshape(-1, 2)
                    tracked_undistorted = np.asarray(
                        package.get('tracked_undistorted_features', []),
                        dtype=np.float32,
                    ).reshape(-1, 2)
                    if not (len(tracked_ids) == len(tracked_raw) == len(tracked_undistorted)):
                        raise ValueError("tracked feature arrays are not aligned")
                    tracked_mask = np.asarray([
                        feature_id not in self.landmark_denylist
                        for feature_id in tracked_ids
                    ], dtype=bool)
                    new_frame.add_tracked_features(
                        tracked_raw[tracked_mask],
                        tracked_undistorted[tracked_mask],
                        tracked_ids[tracked_mask],
                    )
                    new_frame.set_image(image)
                    new_frame.set_is_stationary(is_stationary)

                    self.next_f_id += 1
                    self._schedule_frame(new_frame, is_stationary)

            except queue.Empty:
                self._drain_loop_optimization_results()
                continue
        
        self._export_loop_corrected_trajectory()
        print("【Estimator】thread has finished.")

    def _drain_loop_results(self):
        """Consume inference output without allowing it to mutate the map yet."""
        if self.loop_result_queue is None:
            return
        while True:
            try:
                result = self.loop_result_queue.get_nowait()
            except queue.Empty:
                break
            self.latest_loop_detection = result
            candidates = result.get('candidates', [])
            if candidates:
                best = candidates[0]
                print(
                    f"【LoopDetector】: frame {result['frame_id']} candidate "
                    f"{best['frame_id']} similarity={best['similarity']:.4f} "
                    f"brief_matches={best.get('brief_match_count', 0)} "
                    f"epipolar_inliers={best.get('epipolar_inlier_count', 0)} "
                    f"inlier_ratio={best.get('epipolar_inlier_ratio', 0.0):.2f} "
                    f"inference={result['inference_ms']:.1f} ms"
                )
                # 显示最佳回环检测结果，通过GUI调度器发布（避免线程死锁）
                # result中包含的最优匹配结果：匹配对，当前KF的KLT结果和历史KF的匹配特征点结果
                self._show_best_loop_result(result, best)
                if self.loop_optimization is not None:
                    if self.loop_optimization_busy:
                        # 地图快照必须在真正提交时获取，因此忙时只保留检测结果。
                        self.pending_loop_detection = result
                    else:
                        self._submit_loop_optimization(result)

    def _pose_in_loop_global(self, pose):
        return self.loop_drift @ np.asarray(pose, dtype=np.float64).reshape(4, 4)

    def _record_loop_trajectory_frames(self, frames):
        for frame in frames:
            pose = frame.get_global_pose()
            if pose is None:
                continue
            self.loop_trajectory_records[int(frame.get_id())] = {
                'timestamp': float(frame.get_timestamp()),
                'pose': self._pose_in_loop_global(pose).copy(),
            }

    def _apply_pose_graph_to_trajectory(self, input_poses, optimized_poses):
        correction_ids = sorted(set(input_poses) & set(optimized_poses))
        if not correction_ids:
            return
        corrections = {
            frame_id: (
                np.asarray(optimized_poses[frame_id], dtype=np.float64).reshape(4, 4)
                @ np.linalg.inv(np.asarray(
                    input_poses[frame_id], dtype=np.float64
                ).reshape(4, 4))
            )
            for frame_id in correction_ids
        }
        for frame_id, record in self.loop_trajectory_records.items():
            if frame_id in optimized_poses:
                record['pose'] = np.asarray(
                    optimized_poses[frame_id], dtype=np.float64
                ).reshape(4, 4).copy()
                continue
            index = bisect_right(correction_ids, frame_id) - 1
            correction_id = correction_ids[max(index, 0)]
            record['pose'] = corrections[correction_id] @ record['pose']

    def _export_loop_corrected_trajectory(self):
        if (self.loop_trajectory_exported
                or self.loop_corrected_trajectory_path is None):
            return
        output_path = self.loop_corrected_trajectory_path
        temporary_path = output_path.with_name(output_path.name + '.tmp')
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            records = sorted(
                self.loop_trajectory_records.values(),
                key=lambda record: record['timestamp'],
            )
            with temporary_path.open('w', encoding='utf-8') as handle:
                handle.write("# timestamp tx ty tz qx qy qz qw\n")
                for record in records:
                    pose = record['pose']
                    translation = pose[:3, 3]
                    quaternion = gtsam.Rot3(pose[:3, :3]).toQuaternion()
                    handle.write(
                        f"{record['timestamp']} "
                        f"{translation[0]} {translation[1]} {translation[2]} "
                        f"{quaternion.x()} {quaternion.y()} {quaternion.z()} "
                        f"{quaternion.w()}\n"
                    )
            temporary_path.replace(output_path)
            self.loop_trajectory_exported = True
            print(
                f"【LoopOptimization】: exported {len(records)} corrected poses "
                f"to {output_path}"
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"【LoopOptimization】: corrected trajectory export failed: {exc}")

    def _submit_loop_optimization(self, detection):
        candidates = []
        for candidate in detection.get('candidates', []):
            matches = candidate.get('epipolar_matches', {})
            candidates.append({
                'frame_id': int(candidate['frame_id']),
                'similarity': float(candidate['similarity']),
                'epipolar_matches': {
                    'current_points_3d_camera': np.asarray(
                        matches.get('current_points_3d_camera', []),
                        dtype=np.float64,
                    ).reshape(-1, 3).copy(), # 当前KF坐标系下的3D点
                    'history_undistorted_points': np.asarray(
                        matches.get('history_undistorted_points', []),
                        dtype=np.float64,
                    ).reshape(-1, 2).copy(),
                },
            })
        if not candidates:
            return False

        # 获取全局位姿图
        frame_poses = self.global_map.pose_graph_snapshot()
        for frame in self.local_map.get_active_frames():
            pose = frame.get_global_pose()
            if frame.get_is_keyframe() and pose is not None:
                frame_poses[int(frame.get_id())] = self._pose_in_loop_global(pose)

        current_frame_id = int(detection['frame_id'])
        if current_frame_id not in frame_poses:
            # detector较慢时关键帧应已进入GlobalMap；不存在说明该结果无法安全定位。
            print(
                f"【LoopOptimization】: skip frame {current_frame_id}; "
                "keyframe pose is no longer available."
            )
            return False

        self.loop_optimization_task_id += 1
        job = {
            'task_id': self.loop_optimization_task_id,
            'current_frame_id': current_frame_id,
            'candidates': candidates,
            'frame_poses': frame_poses,
            'drift_at_submission': self.loop_drift.copy(),
        }
        if not self.loop_optimization.submit(job):
            self.pending_loop_detection = detection
            return False
        self.loop_optimization_busy = True
        return True

    def _drain_loop_optimization_results(self, allow_pending=True):
        if self.loop_optimization_result_queue is None:
            return
        while True:
            try:
                result = self.loop_optimization_result_queue.get_nowait()
            except queue.Empty:
                break
            self.loop_optimization_busy = False
            self.latest_loop_optimization = result
            if not result.get('success'):
                print(
                    f"【LoopOptimization】: frame {result.get('current_frame_id')} "
                    f"rejected: {result.get('reason', 'unknown error')}"
                )
                continue

            input_poses = result['input_poses']
            optimized_poses = result['optimized_poses']
            current_frame_id = int(result['current_frame_id'])
            if (current_frame_id not in input_poses
                    or current_frame_id not in optimized_poses):
                print("【LoopOptimization】: invalid PGO result without frontier pose.")
                continue
            correction = (
                optimized_poses[current_frame_id]
                @ np.linalg.inv(input_poses[current_frame_id])
            )
            self.loop_drift = correction @ result['drift_at_submission']

            # 更新全局位姿图
            self.global_map.apply_pose_graph_update(
                input_poses, optimized_poses, self.T_bc
            )

            # 更新回环轨迹记录
            self._apply_pose_graph_to_trajectory(input_poses, optimized_poses)
            print(
                f"【LoopOptimization】: loop {result['history_frame_id']} -> "
                f"{current_frame_id}, PnP {result['pnp_inlier_count']}/"
                f"{result['pnp_input_count']} "
                f"rmse={result['pnp_reprojection_rmse']:.2f}px, "
                f"GTSAM_PGO={result['elapsed_ms']:.1f}ms "
                f"iterations={result['optimizer_nfev']} "
                f"factors={result['optimizer_factor_count']} "
                f"error={result['optimizer_initial_cost']:.3f}->"
                f"{result['optimizer_cost']:.3f}"
            )

        # 如果回环优化任务队列中还有任务，则继续提交
        if allow_pending and not self.loop_optimization_busy:
            pending = self.pending_loop_detection
            self.pending_loop_detection = None
            if pending is not None:
                self._submit_loop_optimization(pending)

    def _finish_loop_optimization(self):
        """At EOF, briefly collect the one in-flight job without risking a hang."""
        self.pending_loop_detection = None
        if not self.loop_optimization_busy:
            return
        deadline = time.monotonic() + self.loop_optimization_shutdown_wait
        while self.loop_optimization_busy and time.monotonic() < deadline:
            self._drain_loop_optimization_results(allow_pending=False)
            if self.loop_optimization_busy:
                time.sleep(0.01)
        if self.loop_optimization_busy:
            print(
                "【LoopOptimization】: final job is still running; "
                "skip waiting to avoid blocking shutdown."
            )

    def _show_best_loop_result(self, result, best):
        """Show the best SALAD candidate and epipolar-RANSAC inliers."""
        if not self.loop_visualize or self.gui_dispatcher is None:
            return
        current_encoded = result.get('current_image_jpeg')
        best_encoded = result.get('best_image_jpeg')
        if current_encoded is None or best_encoded is None:
            return

        try:
            current_image = cv2.imdecode(
                np.frombuffer(current_encoded, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            best_image = cv2.imdecode(
                np.frombuffer(best_encoded, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if current_image is None or best_image is None:
                raise ValueError("failed to decode loop visualization image")

            target_height = min(current_image.shape[0], best_image.shape[0])
            if current_image.shape[0] != target_height:
                current_image = cv2.resize(
                    current_image,
                    (int(round(current_image.shape[1] * target_height /
                               current_image.shape[0])), target_height),
                    interpolation=cv2.INTER_AREA,
                )
            if best_image.shape[0] != target_height:
                best_image = cv2.resize(
                    best_image,
                    (int(round(best_image.shape[1] * target_height /
                               best_image.shape[0])), target_height),
                    interpolation=cv2.INTER_AREA,
                )

            canvas = np.hstack((current_image, best_image))
            matches = best.get('epipolar_matches', {})
            current_points = np.asarray(
                matches.get('current_raw_points', []), dtype=np.float32
            ).reshape(-1, 2)
            history_points = np.asarray(
                matches.get('history_raw_points', []), dtype=np.float32
            ).reshape(-1, 2)
            if len(current_points) != len(history_points):
                raise ValueError("loop match point arrays are not aligned")

            current_source_shape = result.get(
                'current_image_shape', current_image.shape[:2]
            )
            best_source_shape = result.get(
                'best_image_shape', best_image.shape[:2]
            )
            current_source_height, current_source_width = current_source_shape[:2]
            best_source_height, best_source_width = best_source_shape[:2]
            if min(current_source_height, current_source_width,
                   best_source_height, best_source_width) <= 0:
                raise ValueError("invalid source image size for loop visualization")

            current_scale = np.array([
                current_image.shape[1] / float(current_source_width),
                current_image.shape[0] / float(current_source_height),
            ], dtype=np.float32)
            best_scale = np.array([
                best_image.shape[1] / float(best_source_width),
                best_image.shape[0] / float(best_source_height),
            ], dtype=np.float32)
            current_display_points = current_points * current_scale
            history_display_points = history_points * best_scale
            history_display_points[:, 0] += current_image.shape[1]

            for index, (current_point, history_point) in enumerate(zip(
                    current_display_points, history_display_points)):
                if not (np.all(np.isfinite(current_point))
                        and np.all(np.isfinite(history_point))):
                    continue
                color = (
                    int(64 + (37 * index) % 192),
                    int(64 + (73 * index) % 192),
                    int(64 + (109 * index) % 192),
                )
                point_current = tuple(np.rint(current_point).astype(int))
                point_history = tuple(np.rint(history_point).astype(int))
                cv2.line(canvas, point_current, point_history, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, point_current, 3, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, point_history, 3, color, -1, cv2.LINE_AA)

            cv2.putText(
                canvas, f"current: {result['frame_id']}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"best: {best['frame_id']}  score: {best['similarity']:.3f}  "
                f"brief: {best.get('brief_match_count', 0)}  "
                f"E-inliers: {best.get('epipolar_inlier_count', 0)}",
                (current_image.shape[1] + 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA,
            )
            self.gui_dispatcher.publish(
                self.loop_visualization_window,
                canvas,
            )
        except (cv2.error, ValueError) as exc:
            # Invalid display data must not terminate estimator.
            self.loop_visualize = False
            print(f"【LoopDetector】: visualization disabled: {exc}")

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
        self._prune_imu_buffer()

    def _prune_imu_buffer(self):
        """只保留包住当前 VIO 窗口所需的 IMU 样本。

        已边缘化区间的测量保存在预积分缓存里，早于最早活跃帧或待处理帧的原始样本不再需要。
        截止时刻上或之前仍留一条，供边界插值。
        """
        required_timestamps = [
            frame.get_timestamp() for frame in self.local_map.get_active_frames()
        ]
        required_timestamps.extend(
            frame.get_timestamp() for frame, _ in self.scheduled_frames
        )
        if not required_timestamps:
            return
        cutoff = min(required_timestamps)
        while (len(self.imu_buffer) > 2
               and self.imu_buffer[1]['timestamp'] <= cutoff + 1e-9):
            self.imu_buffer.popleft()

    def _schedule_frame(self, frame, is_stationary):
        """图像时刻若还没有 IMU，就先挂起。EuRoC 常有一条与图像时间相同的 IMU，它可能排在图像之后。"""
        self.scheduled_frames.append((frame, is_stationary))
        self._drain_scheduled_frames()

    def _imu_covers_timestamp(self, timestamp):
        """最新一条 IMU 是否已经不早于（晚于）该图像时刻。中值积分需要用它做右端点。"""
        if not self.imu_buffer:
            return False
        return self.imu_buffer[-1]['timestamp'] + 1e-9 >= timestamp

    def _drain_scheduled_frames(self):
        """从队列中取出最早的帧，并将其添加到LocalMap中。"""
        while self.scheduled_frames:
            frame, is_stationary = self.scheduled_frames[0]
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
                self.process_frame_data(frame, is_stationary)
                self.last_processed_f_id = frame.get_id()

    @staticmethod
    def _feature_map(frame):
        return {
            int(feature_id): np.asarray(point, dtype=float).reshape(-1)[:2]
            for feature_id, point in zip(
                frame.get_visual_feature_ids(), frame.get_visual_features())
        }

    def _tracked_points_3d_camera(self, frame):
        """Copy metric landmarks into this keyframe's camera coordinates for async PnP."""
        feature_ids = np.asarray(
            frame.get_tracked_feature_ids(), dtype=np.int64
        ).reshape(-1)
        points = np.full((len(feature_ids), 3), np.nan, dtype=np.float64)
        pose_w_b = frame.get_global_pose()
        if pose_w_b is None:
            return points
        T_wc = np.asarray(pose_w_b, dtype=np.float64).reshape(4, 4) @ self.T_bc
        R_cw = T_wc[:3, :3].T
        t_wc = T_wc[:3, 3]
        for index, feature_id in enumerate(feature_ids):
            landmark = self.local_map.landmarks.get(int(feature_id))
            if landmark is None or landmark.position_3d is None:
                continue
            position_w = np.asarray(
                landmark.position_3d, dtype=np.float64
            ).reshape(3)
            if np.isfinite(position_w).all():
                points[index] = R_cw @ (position_w - t_wc)
        return points

    def judge_keyframe(self, active_frames):
        """判断倒数第二帧是否为关键帧。视差用去畸变像素坐标计算。"""
        if len(active_frames) < 3:
            return True

        # 获取最新帧和次新帧的特征点
        newest_features = self._feature_map(active_frames[-1])
        previous_features = self._feature_map(active_frames[-2])
        tracked_into_newest = newest_features.keys() & previous_features.keys()

        # 最新帧和次新帧的共视太少，确认次新帧为关键帧。
        if len(tracked_into_newest) < self.keyframe_min_tracked:
            return True

        # 次新帧和它的前一帧没有共视时，也作为关键帧。
        frame_left = self._feature_map(active_frames[-3])
        frame_right = previous_features
        common_ids = frame_left.keys() & frame_right.keys()
        if not common_ids:
            return True

        # 这两帧的共视平均像素视差达到阈值，确认次新帧为关键帧。
        parallax = float(np.mean([
            np.linalg.norm(frame_right[feature_id] - frame_left[feature_id])
            for feature_id in common_ids
        ]))
        return parallax >= self.keyframe_parallax

    def marginalization_flag(self, is_keyframe):
        """关键帧边缘化最老帧，普通帧边缘化次新帧。"""
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
        """边缘化成功后删除旧帧观测。host 离开时消除旧逆深度，剩余观测改挂到新 host。"""
        if not isinstance(result, dict) or not result.get('success'):
            return
        frame_id = result.get('marginalized_frame_id')
        if frame_id is None:
            return
        # removed_landmark_ids 只含死亡路标；存活点已转移 host，不在此删除。
        landmark_ids = result.get('removed_landmark_ids', [])
        kept = result.get('kept_landmark_count', 0)
        print(
            f"【Estimator】: Commit marginalization of frame {frame_id}, "
            f"kept {kept} landmarks, removed {len(landmark_ids)} landmarks."
        )
        self._drop_imu_cache_for_frame(frame_id)
        frame = self.local_map.frames.get(frame_id)
        landmark_reference_frames = {
            int(lm_id): int(landmark.host_frame_id)
            for lm_id, landmark in self.local_map.landmarks.items()
        }
        archived_landmarks = self.local_map.landmark_positions(landmark_ids)
        archived_landmarks.update(self.local_map.remove_frame(frame_id, transfer_host=True))
        archived_landmarks = {
            int(lm_id): (
                self.loop_drift[:3, :3]
                @ np.asarray(position, dtype=np.float64).reshape(3)
                + self.loop_drift[:3, 3]
            )
            for lm_id, position in archived_landmarks.items()
        }
        for lm_id in landmark_ids:
            self.local_map.landmarks.pop(lm_id, None)
        if frame is not None and frame.get_global_pose() is not None:
            archived_pose = self._pose_in_loop_global(frame.get_global_pose())
            self.global_map.add_frame(
                frame_id, archived_pose,
                is_keyframe=frame.get_is_keyframe(),
            )
            if frame.get_is_keyframe() and self.loop_detector is not None:
                self.loop_detector.mark_frame_archived(frame_id)
        self.global_map.add_landmarks(
            archived_landmarks,
            reference_frame_ids={
                int(lm_id): landmark_reference_frames.get(int(lm_id), frame_id)
                for lm_id in archived_landmarks
            },
        )
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
        poses = []
        frame_index = {}
        for index, frame in enumerate(frame_window):
            body_pose = frame.get_global_pose()
            poses.append(None if body_pose is None else body_pose @ self.T_bc)
            frame_index[frame.get_id()] = index

        max_error = float(self.config.get('triangulation_max_reprojection_error', 5.0))
        for lm in self.local_map.get_candidate_landmarks():
            observations = [
                (frame_index[frame_id], uv)
                for frame_id, uv in lm.observations.items()
                if frame_id in frame_index and poses[frame_index[frame_id]] is not None
            ]
            if len(observations) < 3:
                continue

            # 多帧三角化新地图点（包含重投影误差、深度范围、视差角检查）
            point = self.sfm_processor.triangulate_track(
                observations, poses, max_error=max_error,
                min_depth=self.local_map.min_depth,
                max_depth=self.local_map.max_depth,
                min_parallax_angle_deg=float(self.config.get('min_parallax_angle_deg', 0.5)))
            if point is None:
                continue
            lm.set_triangulated(point)
            newly_triangulated_for_backend[lm.id] = point

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
        alignment_success, scale, gyro_bias, velocities, refined_gravity, diagnostics = VIOInitializer.initialize(
            initial_keyframes, 
            initial_imu_factors, 
            self.imu_processor, 
            self.gravity_magnitude, 
            self.T_bc
        )

        if alignment_success:
            self.init_alignment_attempts += 1
            accepted, reason = self._accept_initial_alignment(
                scale, gyro_bias, diagnostics, self.init_alignment_attempts)
            print(
                f"【Init Quality】span={diagnostics['span']:.3f}s "
                f"excitation={diagnostics['excitation']:.3f} "
                f"gravity={diagnostics['linear_gravity_norm']:.3f} "
                f"cond={diagnostics['condition_number']:.3e} scale={scale} "
                f"bg={np.asarray(gyro_bias)} {'PASS' if accepted else 'REJECT ' + reason}"
            )
            if not accepted:
                print("【Init】: Quality gate rejected this window. Sliding window.")
                self._discard_oldest_frame()
                return False

            gravity_w = VIOInitializer.align_to_world_frame(
                initial_keyframes, velocities, refined_gravity, scale, self.T_bc)
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
            self._record_loop_trajectory_frames(
                self.local_map.get_active_frames()
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
            
            self._publish_viewer_state("Init")
            # viewer可视化
            
        else:
            print("【Init】: V-I Alignment failed.")
            self._discard_oldest_frame()

        return alignment_success

    def _accept_initial_alignment(self, scale, gyro_bias, diagnostics, attempt_count):
        """Reject observably weak V-I alignment before it mutates window poses."""
        if not self.init_quality_gate:
            return True, 'gate disabled'
        if attempt_count < self.init_min_alignment_attempts:
            self.previous_init_scale = float(scale)
            return False, 'waiting for a neighboring scale'
        if not np.isfinite(scale) or scale <= 0.0:
            return False, 'non-positive scale'
        if diagnostics['span'] < float(self.config.get('init_min_span', 0.8)):
            return False, 'window too short'
        if diagnostics['excitation'] < float(self.config.get('init_min_excitation', 0.25)):
            return False, 'insufficient IMU excitation'
        gravity_error = abs(diagnostics['linear_gravity_norm'] - self.gravity_magnitude)
        if gravity_error > float(self.config.get('init_max_gravity_error', 1.0)):
            return False, 'linear gravity inconsistent'
        if diagnostics['condition_number'] > float(self.config.get('init_max_condition_number', 1e8)):
            return False, 'ill-conditioned alignment'
        if np.linalg.norm(np.asarray(gyro_bias, dtype=float)) > float(
                self.config.get('init_max_gyro_bias', 0.5)):
            return False, 'gyro bias too large'
        if self.previous_init_scale is None:
            self.previous_init_scale = float(scale)
            return False, 'waiting for a neighboring scale'
        relative_scale_change = abs(float(scale) - self.previous_init_scale) / max(
            abs(float(scale)), abs(self.previous_init_scale), 1e-9)
        self.previous_init_scale = float(scale)
        if relative_scale_change > float(self.config.get('init_max_scale_change', 0.20)):
            return False, f'neighboring scale changed by {relative_scale_change:.3f}'
        return True, 'quality checks passed'
    
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

    def process_frame_data(self, new_frame, is_stationary):
        active_frames = self.local_map.get_active_frames()

        # 当前帧到达后，判定倒数第二帧是否为关键帧。
        candidate_frame = active_frames[-2] if len(active_frames) >= 2 else new_frame
        is_keyframe = self.judge_keyframe(active_frames)
        candidate_frame.set_is_keyframe(is_keyframe)

        # 边缘化类型判断
        margin_flag = self.marginalization_flag(is_keyframe)
        margin_name = "MARGIN_OLD" if margin_flag == self.MARGIN_OLD else "MARGIN_SECOND_NEW"
        print(
            f"【Estimator】: Candidate frame {candidate_frame.get_id()} "
            f"is_keyframe={is_keyframe}, triggered_by={new_frame.get_id()} "
            f"margin={margin_name}"
        )

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
        # 4. 完整窗口优化。边缘化类型由次新候选帧是否为关键帧决定。
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
        self._record_loop_trajectory_frames([new_frame])

        # 在后端结果回写后复制候选KF的PnP所需3D点；异步线程之后不再访问LocalMap。
        if is_keyframe and self.loop_detector is not None:
            # 拷贝候选KF追踪的3D点，用于回环PnP计算历史KF的漂移pose
            points_3d_camera = self._tracked_points_3d_camera(candidate_frame)
            if not self.loop_detector.submit_keyframe(
                    candidate_frame, points_3d_camera):
                if not self.loop_submission_warning_printed:
                    print(
                        f"【LoopDetector】: skipped keyframe {candidate_frame.get_id()} "
                        "because the worker is unavailable."
                    )
                    self.loop_submission_warning_printed = True

        # 边缘化最老帧
        if margin_flag == self.MARGIN_OLD:
            observation_frames = {
                lm_id: set(self.local_map.landmarks[lm_id].observations.keys())
                for lm_id in window_landmarks
                if lm_id in self.local_map.landmarks
            }
            margin_result = self.backend.marginalize_oldest(
                window_imu_factors, opt_result['visual_groups'], observation_frames)
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
                    hosted_landmark_ids = [
                        lm_id for lm_id, landmark in self.local_map.landmarks.items()
                        if landmark.host_frame_id == frame_b.get_id()
                    ]
                    margin_result = self.backend.marginalize_second_newest(
                        frame_b.get_id(), hosted_landmark_ids)
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
        
        self._publish_viewer_state("Tracking")

    def _publish_viewer_state(self, stage):
        if not self.viewer_queue:
            return
        active_frames = self.local_map.get_active_frames()
        poses = {
            frame.get_id(): self._pose_in_loop_global(frame.get_global_pose())
            for frame in active_frames if frame.get_global_pose() is not None
        }
        landmarks = {
            landmark_id: (
                self.loop_drift[:3, :3]
                @ np.asarray(position, dtype=np.float64).reshape(3)
                + self.loop_drift[:3, 3]
            )
            for landmark_id, position in self.local_map.get_active_landmarks().items()
        }
        global_poses, global_landmarks = self.global_map.snapshot()
        print(
            f"【Viewer】: {stage} local {len(poses)} poses / {len(landmarks)} landmarks, "
            f"global {len(global_poses)} poses / {len(global_landmarks)} landmarks."
        )
        try:
            self.viewer_queue.put_nowait({
                'poses': poses,
                'landmarks': landmarks,
                'global_poses': global_poses,
                'global_landmarks': global_landmarks,
            })
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
