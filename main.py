import os
import argparse
import sys
import yaml
from pathlib import Path
import queue
import time
import multiprocessing as mp

from utils.dataloader import UnifiedDataloader
from datatype.global_map import GlobalMap
from core.imu_process import IMUProcessor
from core.estimator import Estimator
from core.feature_tracker import FeatureTracker
from core.loop_detector import LoopDetector
from utils.gui_dispatcher import GuiDispatcher
from utils.viewer import Viewer3D

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

    # 前后端队列不限长度。离线 IMU 是 200 Hz，有限队列满了会丢样本。
    feature_tracker_to_estimator_queue = queue.Queue(maxsize=0)
    estimator_to_viewer_queue = queue.Queue(maxsize=20)
    loop_result_queue = queue.Queue(
        maxsize=max(1, int(config.get('loop_detector_result_queue_size', 8)))
    )
    gui_dispatcher = GuiDispatcher() # 创建GUI调度器

    # 初始化数据加载器
    dataloader_config = {
        'path': config['dataset_path'],
        'dataset_type': config['dataset_type'],
        # Kalibr convention: t_imu = t_cam + timeshift_cam_imu.
        # Apply the shift before image/IMU events are sorted and packetized.
        'timeshift_cam_imu': config.get('timeshift_cam_imu', 0.0),
        'apply_timeshift_cam_imu': config.get('apply_timeshift_cam_imu', True),
    }
    data_loader = UnifiedDataloader(dataloader_config)

    # 实例化所有模块
    # SALAD回环检测器（可选）
    loop_detector = (
        LoopDetector(config, loop_result_queue)
        if config.get('enable_loop_detection', False)
        else None
    )
    feature_tracker = FeatureTracker(
        config, data_loader, imu_processor, feature_tracker_to_estimator_queue,
        gui_dispatcher=gui_dispatcher,
    )
    estimator = Estimator(
        config, imu_processor, feature_tracker_to_estimator_queue,
        estimator_to_viewer_queue, global_central_map,
        loop_detector=loop_detector, loop_result_queue=loop_result_queue,
        gui_dispatcher=gui_dispatcher,
    )
    viewer = Viewer3D(estimator_to_viewer_queue) if config.get('enable_viewer', True) else None
    
    # 3. 启动所有线程
    print("Starting all SLAM threads...")
    if loop_detector:
        loop_detector.start()
    feature_tracker.start()
    estimator.start()

    # 启动viewer线程
    if viewer:
        viewer.start()

    try:
        # OpenCV HighGUI is driven exclusively by the main thread. Timed joins
        # leave it free to process window events while workers are alive.
        while feature_tracker.is_alive() or estimator.is_alive():
            gui_dispatcher.show_pending()
            feature_tracker.join(timeout=0.005)
            estimator.join(timeout=0.005)
        gui_dispatcher.show_pending()

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
        if loop_detector:
            loop_detector.shutdown()

        if feature_tracker.is_alive(): feature_tracker.join()
        if estimator.is_alive(): estimator.join()
        if viewer and viewer.is_alive(): viewer.join()
        if loop_detector and loop_detector.is_alive():
            loop_detector.join(timeout=5.0)
            if loop_detector.is_alive():
                print("【LoopDetector】: worker is still loading; leaving daemon thread to exit.")

        gui_dispatcher.close()
        print("[Main Process] SLAM system shut down.")

        sent_imu = feature_tracker.sent_imu_count
        received_imu = estimator.received_imu_count
        print(f"【IMU Queue】sent {sent_imu} received {received_imu}")
        if sent_imu != received_imu:
            print("【IMU Queue】ERROR: sent IMU count does not match received IMU count.")
            sys.exit(1)

if __name__ == "__main__":
    main()
