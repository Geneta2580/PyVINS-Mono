import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Union, Tuple, Optional

@dataclass
class ImuMeasurement:
    accel: Union[np.ndarray, list]
    gyro: Union[np.ndarray, list]

class UnifiedDataloader:
    def __init__(self, dataset_config: dict):
        self.config = dataset_config
        self.base_path = Path(dataset_config['path'])
        self.dataset_type = dataset_config.get('dataset_type', 'euroc').lower()
        
        print(f"【Dataloader】: Initializing for dataset type: {self.dataset_type}")

        # === 1. 根据数据集类型分发处理 ===
        if self.dataset_type == 'euroc':
            self.unified_data = self._load_euroc()
        elif self.dataset_type == 'tum':
            self.unified_data = self._load_tum()
        elif self.dataset_type == 'kitti':
            self.unified_data = self._load_kitti()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        # === 2. 全局排序 ===
        if not self.unified_data.empty:
            self.unified_data.sort_values(by='timestamp', inplace=True, ignore_index=True)
            print(f"【Dataloader】: Dataloader initialized with {len(self.unified_data)} total events")
        else:
            print("【Warning】: Dataframe is empty!")

        self.current_idx = 0

    def _load_euroc(self) -> pd.DataFrame:
        """读取EuRoC格式数据 (原始单位: 纳秒)"""
        cam0_csv_path = self.base_path / 'mav0' / 'cam0' / 'data.csv' 
        
        if not cam0_csv_path.exists():
             cam0_csv_path = self.base_path / 'image' / 'data.csv'
             self.cam_data_path = self.base_path / 'image' / 'data'
             imu0_csv_path = self.base_path / 'imu' / 'data.csv'
        else:
             self.cam_data_path = self.base_path / 'mav0' / 'cam0' / 'data'
             imu0_csv_path = self.base_path / 'mav0' / 'imu0' / 'data.csv'

        # 读取相机
        cam0_df = pd.read_csv(cam0_csv_path, header=0, names=['timestamp', 'filename'])
        cam0_df['type'] = 'IMAGE'
        cam0_df['filepath'] = cam0_df['filename'].apply(lambda x: str(self.cam_data_path / x))
        
        # 【时间戳转换】 EuRoC是纳秒整数 -> 秒(UTC浮点)
        cam0_df['timestamp'] = cam0_df['timestamp'] * 1e-9

        # 读取IMU
        try:
            imu0_df = pd.read_csv(
                imu0_csv_path, 
                header=0, 
                names=['timestamp', 'w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']
            )
            
            imu0_df['type'] = 'IMU'
            # 【时间戳转换】 EuRoC是纳秒整数 -> 秒(UTC浮点)
            imu0_df['timestamp'] = imu0_df['timestamp'] * 1e-9
            
            print(f"【Dataloader】: Loaded {len(cam0_df)} images and {len(imu0_df)} IMU msgs (EuRoC)")
            return pd.concat([cam0_df, imu0_df])
        except FileNotFoundError:
            print(f"【Warning】: IMU data not found at {imu0_csv_path}, ensuring pure vision.")
            return cam0_df

    def _load_tum(self) -> pd.DataFrame:
        """读取TUM RGB-D格式数据 (原始单位: 秒)"""
        cam0_csv_path = self.base_path / 'rgb.txt'
        self.cam_data_path = self.base_path 

        cam0_df = pd.read_csv(
            cam0_csv_path, comment='#', header=None, sep=' ', names=['timestamp', 'filename']
        )
        cam0_df['type'] = 'IMAGE'
        cam0_df['filepath'] = cam0_df['filename'].apply(lambda x: str(self.cam_data_path / x))
        
        # TUM 本身就是秒，通常无需处理，但有些版本可能是字符串
        cam0_df['timestamp'] = cam0_df['timestamp'].astype(float)
        
        print(f"【Dataloader】: Loaded {len(cam0_df)} images (TUM). IMU disabled.")
        return cam0_df

    def _load_kitti(self) -> pd.DataFrame:
        """
        读取KITTI Raw数据，并将时间戳强制转为 UTC 秒。
        """

        # --- 1. UTC 时间戳解析函数 ---
        def parse_kitti_ts_utc(ts_str):
            """
            将 KITTI 时间字符串 "YYYY-MM-DD HH:MM:SS.ssssss" 转换为 UTC 时间戳
            """
            ts_str = ts_str.strip()
            try:
                # 尝试解析带日期的完整格式
                # 截断到26位以适配datetime (Python只支持6位微秒)
                ts_str_clean = ts_str[:26] 
                
                # 手动分离纳秒部分以保留精度
                if '.' in ts_str:
                    main_part, frac_part = ts_str.split('.')
                    # 补全或截断小数部分
                    frac_seconds = float("0." + frac_part)
                else:
                    main_part = ts_str
                    frac_seconds = 0.0
                
                # 解析主时间部分
                dt_obj = datetime.strptime(main_part, "%Y-%m-%d %H:%M:%S")
                
                # 【核心】强制设定为 UTC 时区
                # 这样 dt_obj.timestamp() 会直接返回对应的 UNIX 时间，不会减去本地时区偏移
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                
                return dt_obj.timestamp() + frac_seconds

            except ValueError:
                # 如果已经是浮点数字符串
                return float(ts_str)

        # --- 2. 读取图像部分 ---
        timestamp_path = self.base_path / 'image_00' / 'timestamps.txt'
        if not timestamp_path.exists():
            timestamp_path = self.base_path / 'times.txt'

        if not timestamp_path.exists():
            raise FileNotFoundError(f"KITTI timestamps not found.")

        # 使用 UTC 解析函数
        with open(timestamp_path, 'r') as f:
            image_times = [parse_kitti_ts_utc(line) for line in f.readlines()]

        cam_df = pd.DataFrame({'timestamp': image_times})
        cam_df['type'] = 'IMAGE'

        image_dir = self.base_path / 'image_00'
        if not image_dir.exists(): image_dir = self.base_path / 'image_02'

        if (image_dir / "000000.png").exists():
                image_filenames = [str(image_dir / f"{i:06d}.png") for i in range(len(image_times))]
        else:
                image_filenames = [str(image_dir / "data" / f"{i:010d}.png") for i in range(len(image_times))]
        
        cam_df['filepath'] = image_filenames

        # --- 3. 读取并封装 IMU 数据 ---
        oxts_root = self.base_path / 'oxts'
        oxts_time_path = oxts_root / 'timestamps.txt'
        oxts_data_dir = oxts_root / 'data'

        if oxts_time_path.exists() and oxts_data_dir.exists():
            print("【Dataloader】: Found OXTS data, loading IMU...")
            
            # 使用 UTC 解析函数
            with open(oxts_time_path, 'r') as f:
                imu_times = [parse_kitti_ts_utc(line) for line in f.readlines()]
            
            imu_data_list = []
            
            for i, ts in enumerate(imu_times):
                file_path = oxts_data_dir / f"{i:010d}.txt"
                
                if file_path.exists():
                    try:
                        raw_vals = np.fromfile(file_path, sep=' ')
                        
                        # KITTI IMU data extraction
                        accel = raw_vals[11:14]
                        gyro = raw_vals[17:20]
                        
                        imu_vector = np.concatenate((gyro, accel))
                        
                        imu_data_list.append({
                            'timestamp': ts,
                            'type': 'IMU',
                            'data': imu_vector
                        })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
            
            imu_df = pd.DataFrame(imu_data_list)
            
            combined_df = pd.concat([cam_df, imu_df], ignore_index=True)
            combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
            
            return combined_df

        return cam_df

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.unified_data):
            raise StopIteration

        event = self.unified_data.iloc[self.current_idx]
        self.current_idx += 1
        
        event_type = event['type']
        timestamp = event['timestamp']

        if event_type == 'IMAGE':
            img_path = event['filepath']
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"【Warning】: Failed to load image: {img_path}")
                return self.__next__()
            
            return timestamp, event_type, (image, img_path)
        
        elif event_type == 'IMU':
            if 'data' in event and isinstance(event['data'], np.ndarray):
                imu_values = event['data']
            else:
                imu_values = event[['w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']].values.astype(float)
            
            return timestamp, event_type, imu_values