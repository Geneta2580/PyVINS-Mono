#!/usr/bin/env python3
"""IMU from the tracker must reach the estimator with no dropped samples."""

import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.estimator import Estimator
from core.feature_tracker import FeatureTracker
from core.imu_process import IMUProcessor
from utils.dataloader import ImuMeasurement


DATASET = Path("/home/geneta/dataset/source_dataset/euroc/MH_01_easy")
GAP_LIMIT = 0.0075


def make_tracker(output_queue):
    tracker = FeatureTracker.__new__(FeatureTracker)
    tracker.output_queue = output_queue
    tracker.pending_imu = []
    tracker.held_visual = None
    tracker.sent_imu_count = 0
    tracker._shutdown_sent = False
    tracker.use_imu_flow_prediction = False
    tracker.imu_buffer = []
    return tracker


def make_estimator():
    estimator = Estimator.__new__(Estimator)
    estimator.imu_buffer = []
    estimator.received_imu_count = 0
    estimator.use_imu_output = False
    estimator.scheduled_frames = []
    estimator.last_processed_imu_timestamp = -1
    estimator.imu_processor = IMUProcessor({"gravity": 9.81, "max_imu_dt": 0.0125})
    return estimator


def measurement_at(index):
    measurement = ImuMeasurement(
        gyro=np.array([0.0, 0.0, 0.01 * index]),
        accel=np.array([0.0, 0.0, 9.81]),
    )
    return measurement


def load_euroc_events(dataset_dir):
    """Same image-then-IMU concat and timestamp sort as UnifiedDataloader."""
    camera = pd.read_csv(dataset_dir / "mav0" / "cam0" / "data.csv")
    imu = pd.read_csv(dataset_dir / "mav0" / "imu0" / "data.csv")
    camera = camera.iloc[:, :1].copy()
    imu = imu.iloc[:, :1].copy()
    camera.columns = ["timestamp"]
    imu.columns = ["timestamp"]
    camera["timestamp"] = camera["timestamp"].astype(np.int64) * 1e-9
    imu["timestamp"] = imu["timestamp"].astype(np.int64) * 1e-9
    camera["type"] = "IMAGE"
    imu["type"] = "IMU"
    merged = pd.concat([camera, imu], ignore_index=True)
    merged = merged.sort_values(by="timestamp").reset_index(drop=True)
    return list(zip(merged["timestamp"].tolist(), merged["type"].tolist()))


def consume(output_queue, packets, delay):
    while True:
        packet = output_queue.get()
        if delay:
            time.sleep(delay)
        packets.append(packet)
        if packet is None:
            return


def replay(events, maxsize, delay):
    output_queue = queue.Queue(maxsize=maxsize)
    tracker = make_tracker(output_queue)
    packets = []
    consumer = threading.Thread(target=consume, args=(output_queue, packets, delay))
    consumer.start()
    for index, (timestamp, event_type) in enumerate(events):
        if event_type == "IMU":
            tracker._stage_imu(timestamp, measurement_at(index))
        else:
            tracker._stage_visual({"timestamp": timestamp, "visual_features": []})
    tracker._finish_stream()
    consumer.join()
    return tracker, packets


def check_delivery(name, events):
    tracker, packets = replay(events, maxsize=1, delay=0.0002)
    estimator = make_estimator()
    image_times = []
    for packet in packets:
        if packet is None:
            continue
        imu_batch = packet.get("imu_since_last_image")
        if imu_batch:
            estimator._ingest_imu_batch(imu_batch)
        if "visual_features" in packet:
            image_times.append(packet["timestamp"])
            if imu_batch[-1][0] + 1e-9 < packet["timestamp"]:
                raise AssertionError(f"{name}: image packet is missing its right-endpoint IMU")

    sent = tracker.sent_imu_count
    received = estimator.received_imu_count
    expected = sum(event_type == "IMU" for _, event_type in events)
    if not (sent == received == expected):
        raise AssertionError(f"{name}: sent {sent} received {received} dataset {expected}")

    timestamps = [item["timestamp"] for item in estimator.imu_buffer]
    if len(timestamps) >= 2:
        max_dt = float(np.max(np.diff(timestamps)))
        if max_dt > GAP_LIMIT:
            raise AssertionError(f"{name}: IMU gap {max_dt:.6f}s exceeds {GAP_LIMIT}")
    else:
        max_dt = 0.0

    cannot_bracket = 0
    max_bounded_dt = 0.0
    for start_time, end_time in zip(image_times, image_times[1:]):
        bounded = estimator._bounded_imu_between(start_time, end_time)
        if not bounded:
            cannot_bracket += 1
            continue
        bounded_dt = max(bounded[index + 1][0] - bounded[index][0] for index in range(len(bounded) - 1))
        max_bounded_dt = max(max_bounded_dt, bounded_dt)
    if cannot_bracket:
        raise AssertionError(f"{name}: Cannot bracket = {cannot_bracket}")
    if max_bounded_dt > GAP_LIMIT:
        raise AssertionError(f"{name}: bounded IMU gap {max_bounded_dt:.6f}s exceeds {GAP_LIMIT}")

    print(
        f"{name}: sent {sent} received {received} "
        f"images {len(image_times)} cannot_bracket {cannot_bracket} "
        f"max_dt {max_dt:.6f}s max_bounded_dt {max_bounded_dt:.6f}s"
    )


def synthetic_coincident_imu():
    """Image timestamp equals an IMU sample, and that IMU is ordered after the image."""
    events = []
    start = 1000.0
    for step in range(40):
        image_time = start + step * 0.05
        events.append((image_time, "IMAGE"))
        for sample in range(10):
            events.append((image_time + sample * 0.005, "IMU"))
    events.append((start + 40 * 0.05, "IMU"))
    return events


def main():
    check_delivery("synthetic image-before-imu", synthetic_coincident_imu())
    if not DATASET.exists():
        raise SystemExit(f"dataset not found: {DATASET}")
    check_delivery("MH_01_easy", load_euroc_events(DATASET))
    print("TEST_OK")


if __name__ == "__main__":
    main()
