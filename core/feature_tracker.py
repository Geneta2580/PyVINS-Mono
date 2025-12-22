import numpy as np
import threading
from collections import deque
from enum import Enum, auto
import queue

from utils.dataloader import ImuMeasurement
from core.visual_process import VisualProcessor
from core.imu_process import IMUProcessor

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
                
                # 光流追踪特征点
                undistorted_features, feature_ids, is_kf_visual, is_stationary = self.visual_processor.track_features(image_data)

                is_kf = is_kf_visual
                # 判断关键帧逻辑3：时间条件
                if self.last_kf_timestamp is not None:
                    # 间隔大于最大关键帧间隔，强制插入关键帧
                    is_kf_time_max = (timestamp - self.last_kf_timestamp) > self.config.get('max_kf_interval', 5)
                    is_kf_time_min = (timestamp - self.last_kf_timestamp) > self.config.get('min_kf_interval', 0.2)
                    # 视觉条件满足且间隔大于最小关键帧间隔才能插入关键帧
                    is_kf_temp = is_kf_visual and is_kf_time_min
                    is_kf = is_kf_temp or is_kf_time_max

                # 处理图像信息
                visual_features = {
                    'visual_features': undistorted_features,
                    'feature_ids': feature_ids,
                    'timestamp': timestamp,
                    'image': image_data,
                    'is_kf': is_kf,
                    'is_stationary': is_stationary
                }

                try:
                    self.output_queue.put(visual_features, timeout=0.1) 
                except queue.Full:
                    pass

                if is_kf:
                    self.last_kf_timestamp = timestamp
                    print(f"【FeatureTracker】Keyframe: {visual_features['timestamp']}")
        
        # 从数据循环中跳出，表示程序需要结束
        try:
            self.output_queue.put(None, timeout=0.1) 
        except queue.Full:
            pass
        self.is_running = False
        print("Visual Feature Tracker has finished processing all data.")