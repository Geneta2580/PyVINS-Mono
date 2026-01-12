import os
import argparse
import yaml
from pathlib import Path
import queue
import time
import multiprocessing as mp

import cv2

from utils.dataloader import UnifiedDataloader
from datatype.global_map import GlobalMap
from core.imu_process import IMUProcessor
from core.estimator import Estimator
from core.feature_tracker import FeatureTracker
from core.viewer import Viewer3D

def main():
    # 1. 配置参数解析
    parser = argparse.ArgumentParser(description="PyVINS-Fusion: Visual-Inertial SLAM System For Python")

    # 配置文件路径
    parser.add_argument('--config', type=Path, default='config/config_kitti.yaml',
                        help="Path to the configuration file")
    args = parser.parse_args()

    # 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            print("Configuration loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return

    # 2. 初始化模块和通信管道
    print("Initializing SLAM components and communication queues...")
    
    # 启动一个spawn主进程
    mp.set_start_method('spawn', force=True)

    # 创建全局地图/全局积分器
    global_central_map = GlobalMap()
    imu_processor = IMUProcessor(config)

    # 创建队列
    feature_tracker_to_estimator_queue = queue.Queue(maxsize=20)
    estimator_to_viewer_queue = queue.Queue(maxsize=20)

    # 初始化数据加载器
    dataloader_config = {'path': config['dataset_path'], 'dataset_type': config['dataset_type']}
    data_loader = UnifiedDataloader(dataloader_config)

    # 实例化所有模块
    feature_tracker = FeatureTracker(config, data_loader, imu_processor, feature_tracker_to_estimator_queue)
    estimator = Estimator(config, imu_processor, feature_tracker_to_estimator_queue, estimator_to_viewer_queue, global_central_map)
    viewer = Viewer3D(estimator_to_viewer_queue) if config.get('enable_viewer', True) else None
    
    # 3. 启动所有线程
    print("Starting all SLAM threads...")
    feature_tracker.start()
    estimator.start()

    # 启动viewer线程
    if viewer:
        viewer.start()

    try:
        feature_tracker.join()
        estimator.join()

    except KeyboardInterrupt:
        print("\n[Main Process] Caught KeyboardInterrupt, initiating shutdown...")
    finally:
        # 5. 安全关闭所有模块
        print("[Main Process] Shutting down all components...")
        
        # a. 首先，向所有子任务发送停止信号
        feature_tracker.shutdown()
        estimator.shutdown()
        if viewer:
            viewer.shutdown()

        if feature_tracker.is_alive(): feature_tracker.join()
        if estimator.is_alive(): estimator.join()
        if viewer and viewer.is_alive(): viewer.join()

        cv2.destroyAllWindows()
        print("[Main Process] SLAM system shut down.")

if __name__ == "__main__":
    main()