import numpy as np
import threading
from collections import deque
from enum import Enum, auto
import queue

from utils.dataloader import ImuMeasurement
from core.visual_process import VisualProcessor
from core.imu_process import IMUProcessor
from utils.debug import Debugger

class FeatureTracker(threading.Thread):
    def __init__(self, config, dataloader, output_queue):
        super().__init__(daemon=True)
        self.config = config
        self.dataloader = dataloader
        self.output_queue = output_queue

        self.imu_buffer = deque()
        self.last_kf_timestamp = None

        # Threading control
        self.is_running = False

        # 视觉处理模块
        self.visual_processor = VisualProcessor(config)

        # 日志记录（从VisualProcessor移到这里）
        log_columns = [
            "timestamp", "feature_count", "long_track_ratio", "mean_parallax", "is_kf", "is_stationary",
            "is_kf_visual", "is_kf_time", "is_kf_final",
        ]
        self.logger = Debugger(self.config, file_prefix="feature_tracker", column_names=log_columns)

    def start(self):
        self.is_running = True
        super().start()

    def shutdown(self):
        self.is_running = False
        # join操作由主线程负责，这里只设置标志
        print("Visual Feature Tracker shut down signal sent.")

    def run(self):
        print("Visual Feature Tracker thread started.")
        for i, (timestamp, event_type, data) in enumerate(self.dataloader):

            # 如果frontend被关闭，则退出循环
            if not self.is_running:
                break

            # 处理IMU数据
            if event_type == 'IMU':
                # 注意角速度在前，加速度在后
                data = ImuMeasurement(gyro = data[0:3], accel = data[3:6])

                if data:
                    imu_measurements = {
                        'imu_measurements': data,
                        'timestamp': timestamp,
                    }
                try:
                    self.output_queue.put(imu_measurements, timeout=0.1)
                except queue.Full:
                    pass

            # 处理图像数据
            elif event_type == 'IMAGE':
                # data[0]是图像数据，data[1]是图像路径      
                image_data = data[0]
                print(f"【FeatureTracker】Image data: {data[1]}")
                
                # 光流追踪特征点（返回stats和viz_payload）
                undistorted_features, feature_ids, stats, viz = self.visual_processor.track_features(image_data, timestamp)

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
                        stats["long_track_ratio"]
                    )

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

                try:
                    self.output_queue.put(visual_features, timeout=0.1) 
                except queue.Full:
                    pass

                if is_kf_final:
                    self.last_kf_timestamp = timestamp
                    print(f"【FeatureTracker】Keyframe: {visual_features['timestamp']}")
        
        # 从数据循环中跳出，表示程序需要结束
        try:
            self.output_queue.put(None, timeout=0.1) 
        except queue.Full:
            pass
        self.is_running = False
        print("Visual Feature Tracker has finished processing all data.")