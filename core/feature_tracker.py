import numpy as np
import threading
from collections import deque
from enum import Enum, auto

from utils.dataloader import ImuMeasurement
from core.visual_process import VisualProcessor
from utils.debug import Debugger
from utils.performance_stats import PerformanceStats

class FeatureTracker(threading.Thread):
    def __init__(self, config, dataloader, imu_processor, output_queue):
        super().__init__(daemon=True)
        self.config = config
        self.dataloader = dataloader
        self.output_queue = output_queue

        self.imu_processor = imu_processor
        self.use_imu_flow_prediction = self.config.get('use_imu_flow_prediction', True)
        self.imu_buffer = deque()
        self.pending_imu = []
        self.held_visual = None
        self.sent_imu_count = 0
        self._shutdown_sent = False
        self.last_image_timestamp = None
        self.last_kf_timestamp = None

        # Threading control
        self.is_running = False

        # 视觉处理模块
        self.visual_processor = VisualProcessor(config)

        # 日志记录（从VisualProcessor移到这里）
        log_columns = [
            "timestamp", "feature_count", "long_track_ratio", "mean_parallax", "is_kf", "is_stationary",
            "is_kf_visual", "is_kf_time", "is_kf_final",
            "instant_fps", "avg_fps",
        ]
        self.logger = Debugger(self.config, file_prefix="feature_tracker", column_names=log_columns)

        self.enable_fps_stats = self.config.get('enable_fps_stats', True)
        self.fps_report_interval = self.config.get('fps_report_interval', 0)
        self.fps_stats = PerformanceStats(name="frontend")
        self._fps_summary_printed = False
        self._logger_closed = False

    def start(self):
        self.is_running = True
        super().start()

    def _report_fps_summary(self):
        if (self.enable_fps_stats
                and self.fps_stats.frame_count > 0
                and not self._fps_summary_printed):
            print(self.fps_stats.format_summary())
            self._fps_summary_printed = True

    def _close_logger(self):
        if not self._logger_closed and hasattr(self.logger, 'close'):
            self.logger.close()
            self._logger_closed = True

    def shutdown(self):
        self.is_running = False
        self._report_fps_summary()
        self._close_logger()
        print("Visual Feature Tracker shut down signal sent.")

    def _put_lossless(self, packet):
        """阻塞直到入队。队列不丢 IMU、图像和结束信号。"""
        self.output_queue.put(packet)

    def _buffer_imu_for_flow(self, timestamp, measurement):
        if not self.use_imu_flow_prediction:
            return
        self.imu_buffer.append((timestamp, measurement))
        max_buffer_time = 1.0
        while len(self.imu_buffer) > 0 and timestamp - self.imu_buffer[0][0] > max_buffer_time:
            self.imu_buffer.popleft()

    def _stage_imu(self, timestamp, measurement):
        """先攒着，等这张图像的右端点 IMU 到齐后随图像一次送出。"""
        self.pending_imu.append((timestamp, measurement))
        self._release_held_visual()

    def _stage_visual(self, visual_features):
        """先挂起这张图像。右端点 IMU 已经在缓冲里就立刻送出。"""
        if self.held_visual is not None:
            raise RuntimeError(
                "previous image is still waiting for an IMU sample at or after its timestamp"
            )
        self.held_visual = visual_features
        self._release_held_visual()

    def _release_held_visual(self):
        """最新 IMU 不早于图像时刻时，连同这段 IMU 一次送出。"""
        if self.held_visual is None or not self.pending_imu:
            return
        image_timestamp = self.held_visual['timestamp']
        if self.pending_imu[-1][0] + 1e-9 < image_timestamp:
            return
        imu_batch = self.pending_imu
        self.pending_imu = []
        packet = self.held_visual
        packet['imu_since_last_image'] = imu_batch
        self.held_visual = None
        self._put_lossless(packet)
        self.sent_imu_count += len(imu_batch)

    def _finish_stream(self):
        if self._shutdown_sent:
            return
        if self.held_visual is not None:
            image_timestamp = self.held_visual['timestamp']
            raise RuntimeError(
                f"image at {image_timestamp:.9f} has no IMU sample at or after its timestamp"
            )
        if self.pending_imu:
            self._put_lossless({'imu_since_last_image': self.pending_imu})
            self.sent_imu_count += len(self.pending_imu)
            self.pending_imu = []
        self._put_lossless(None)
        self._shutdown_sent = True

    def run(self):
        print("Visual Feature Tracker thread started.")
        try:
            for i, (timestamp, event_type, data) in enumerate(self.dataloader):

                if not self.is_running:
                    break

                if event_type == 'IMU':
                    measurement = ImuMeasurement(gyro=data[0:3], accel=data[3:6])
                    self._buffer_imu_for_flow(timestamp, measurement)
                    self._stage_imu(timestamp, measurement)
                    continue

                if event_type != 'IMAGE':
                    continue

                # 处理图像数据
                # data[0]是图像数据，data[1]是图像路径      
                image_data = data[0]
                print(f"【FeatureTracker】Image data: {data[1]}")
                
                # 准备IMU数据用于光流初值预测
                imu_data_for_prediction = None
                if (self.use_imu_flow_prediction
                        and self.last_image_timestamp is not None
                        and len(self.imu_buffer) > 0):
                    # 获取从上一帧到当前帧的IMU测量数据
                    imu_measurements = [
                        (ts, imu_data) for ts, imu_data in self.imu_buffer
                        if self.last_image_timestamp < ts <= timestamp
                    ]
                    if len(imu_measurements) > 0:
                        imu_data_for_prediction = {
                            'measurements': imu_measurements,
                            'start_time': self.last_image_timestamp,
                            'end_time': timestamp
                        }
                
                # 光流追踪特征点（返回stats和viz_payload）
                undistorted_features, feature_ids, stats, viz = self.visual_processor.track_features(
                    image_data, timestamp,
                    imu_data_for_prediction=imu_data_for_prediction,
                    imu_processor=self.imu_processor if self.use_imu_flow_prediction else None,
                )
                
                # 更新上一帧图像时间戳
                self.last_image_timestamp = timestamp

                # 视觉判定（来自VisualProcessor）
                is_kf_visual = int(stats["is_kf_visual"])

                # 时间判定和最终判定
                is_kf_time = 0
                is_kf_final = is_kf_visual
                if self.last_kf_timestamp is not None:
                    dt = timestamp - self.last_kf_timestamp
                    is_kf_time_max = int(dt > self.config.get('max_kf_interval', 5))
                    is_kf_time_min = int(dt > self.config.get('min_kf_interval', 0.2))
                    # 视觉条件满足且间隔大于最小关键帧间隔才能插入关键帧，或者超过最大间隔强制插入
                    is_kf_final = int((is_kf_visual and is_kf_time_min) or is_kf_time_max)
                    is_kf_time = int(is_kf_time_max or is_kf_time_min)
                else:
                    # 第一帧：视觉判定就是最终判定
                    is_kf_final = int(is_kf_visual)
                    is_kf_time = 0

                is_stationary = int(stats["is_stationary"])

                # 可视化：使用最终is_kf_final，并接收返回的vis_img
                vis_img = None
                if self.visual_processor.visualize_flag:
                    vis_img = self.visual_processor.visualize_tracking(
                        image_data,
                        viz["good_prev"], viz["good_curr"], viz["good_ids"],
                        is_kf_final, is_stationary,
                        stats["mean_parallax"],
                        timestamp,
                        stats["prev_total_count"],
                        stats["long_track_ratio"],
                        viz.get("processed_gray"),
                    )

                if self.enable_fps_stats:
                    self.fps_stats.tick()
                    instant_fps = self.fps_stats.instant_fps
                    avg_fps = self.fps_stats.get_average_fps()
                    if (self.fps_report_interval > 0
                            and self.fps_stats.frame_count % self.fps_report_interval == 0):
                        print(
                            f"【Performance】frame #{self.fps_stats.frame_count}: "
                            f"instant={instant_fps:.1f} fps, avg={avg_fps:.1f} fps"
                        )
                else:
                    instant_fps = 0.0
                    avg_fps = 0.0

                # 写入日志：保留原有字段（is_kf保持视觉判定语义），新增3个is_kf字段
                self.logger.log_state({
                    "timestamp": float(stats["timestamp"]),
                    "feature_count": int(stats["feature_count"]),
                    "long_track_ratio": float(stats["long_track_ratio"]),
                    "mean_parallax": float(stats["mean_parallax"]),
                    "is_kf": int(is_kf_visual),              # 保持原"视觉is_kf"的语义
                    "is_stationary": int(is_stationary),

                    "is_kf_visual": int(is_kf_visual),
                    "is_kf_time": int(is_kf_time),
                    "is_kf_final": int(is_kf_final),
                    "instant_fps": float(instant_fps),
                    "avg_fps": float(avg_fps),
                })

                # 处理图像信息
                visual_features = {
                    'visual_features': undistorted_features,
                    'feature_ids': feature_ids,
                    'timestamp': timestamp,
                    'image': image_data,
                    'is_kf': is_kf_final,
                    'is_stationary': is_stationary,
                    'vis_img': vis_img,  # visualize_tracking的返回值
                }

                self._stage_visual(visual_features)

                if is_kf_final:
                    self.last_kf_timestamp = timestamp
                    print(f"【FeatureTracker】Keyframe: {visual_features['timestamp']}")
        finally:
            try:
                self._finish_stream()
            except Exception as exc:
                print(f"【IMU Queue】ERROR: {exc}")
                if not self._shutdown_sent:
                    self._put_lossless(None)
                    self._shutdown_sent = True
            self.is_running = False
            self._report_fps_summary()
            self._close_logger()
            print(f"【IMU Queue】sent {self.sent_imu_count}")
            print("Visual Feature Tracker has finished processing all data.")