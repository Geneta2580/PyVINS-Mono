import numpy as np
import gtsam
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import csv

class Debugger:
    def __init__(self, config, file_prefix="debug", column_names=None):
        if column_names is None:
            column_names = ["timestamp", "value"]
        
        self.column_names = list(column_names) # 使用传入的列名

        # 输出文件地址, 格式为: output/file_prefix_timestamp.csv
        log_dir = config.get('log_dir')
        use_timestamp = True # 是否在文件名中添加时间戳
        self.log_path = self._initialize_log_file(file_prefix, log_dir, use_timestamp)
        self.log_file = open(self.log_path, 'w', newline='')

        # 打开文件并写入表头
        self.writer = csv.DictWriter(self.log_file, fieldnames=self.column_names)
        self.writer.writeheader()
        self.log_file.flush()
        print(f"【Debugger】: Log file initialized at {self.log_path} with columns {self.column_names}")

    def _initialize_log_file(self, prefix, log_dir, use_timestamp):
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{prefix}_{timestamp}.csv"
        else:
            file_name = f"{prefix}.csv"
        
        log_path = os.path.join(log_dir, file_name)

        os.makedirs(log_dir, exist_ok=True)
        
        # --- 3. 打开文件并写入表头 ---
        # 使用'w'模式（写入）和 newline='' 来防止写入空行
        self.file_handle = open(log_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file_handle)
        
        # 写入完整的表头，第一列总是'Round'
        header = self.column_names
        self.writer.writerow(header)
        
        print(f"【Logger】Logging to {log_path}")

        return log_path

    def log(self, value):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.writer.writerow([timestamp, value])
        self.log_file.flush()

    def log_state(self, state_dict):
        """
        Logs a dictionary of state variables. Keys must match column names.
        """
        # 确保所有列都有值，没有的填空字符串
        row_data = {key: state_dict.get(key, '') for key in self.column_names}
        self.writer.writerow(row_data)
        self.log_file.flush()

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            print(f"【Logger】Closed log file {self.log_path}")
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def save_full_matrix_python(matrix):
        try:
            filename = "hessian.txt"
            # 使用 np.savetxt 直接保存
            # delimiter=',' 指定了使用逗号作为分隔符
            # fmt='%.8g' 是一个通用的数字格式，可以保留足够的精度，同时保持格式整洁
            np.savetxt(filename, matrix, delimiter=',', fmt='%.8g')
            
            print(f"Python: 矩阵已成功以 txt 格式写入到 '{filename}'")

        except Exception as e:
            print(f"Python: 写入文件时出错: {e}")

    @staticmethod
    def visualize_matrix(matrix, title="Matrix Heatmap", save_path=None):
        try:
            # 创建一个图形窗口
            plt.figure(figsize=(12, 10))
            
            # 使用 seaborn 的 heatmap 函数
            # cmap='viridis' 是一个视觉上很舒服的色谱
            # annot=False 因为矩阵太大，不适合显示数字
            # square=True 保证每个单元格是正方形
            sns.heatmap(matrix, cmap='viridis', annot=False, square=True)
            
            plt.title(title, fontsize=16)
            plt.xlabel("State Variables", fontsize=12)
            plt.ylabel("State Variables", fontsize=12)
            
            if save_path:
                plt.savefig(save_path)
                print(f"【Debug】: Heatmap saved to {save_path}")
            
            # 显示图形
            # plt.show()

        except Exception as e:
            print(f"【Debug】: Failed to visualize matrix. Error: {e}")


    @staticmethod
    def initialize_trajectory_file(output_path):
        """
        打开一个用于写入轨迹的文件，并写入TUM格式的头部信息。
        如果成功，返回文件句柄；如果发生错误，则返回None。
        """
        try:
            file_handle = open(output_path, 'w')
            # 写入TUM格式的头部信息
            file_handle.write("# timestamp tx ty tz qx qy qz qw\n")
            print(f"【Debugger】成功初始化轨迹文件: {output_path}")
            return file_handle
        except IOError as e:
            print(f"【Debugger】错误: 无法打开轨迹文件 {output_path} 进行写入: {e}")
            return None

    @staticmethod
    def log_trajectory_tum(file_handle, keyframe):
        """
        将单个关键帧的位姿以TUM格式记录到指定的轨迹文件中。
        这是一个静态函数，不依赖于类的实例。

        参数:
        file_handle (File): 用于写入的文件句柄。
        keyframe (KeyFrame): 包含位姿和时间戳的关键帧对象。
        """
        if not file_handle:
            return

        timestamp = keyframe.get_timestamp()
        pose_matrix = keyframe.get_global_pose()

        if pose_matrix is not None:
            # 提取平移向量
            t = pose_matrix[:3, 3]
            # 提取旋转矩阵并转换为gtsam四元数
            rot_matrix = pose_matrix[:3, :3]
            q = gtsam.Rot3(rot_matrix).toQuaternion()
            
            # 按照TUM格式写入: timestamp tx ty tz qx qy qz qw
            # gtsam.Quaternion 的顺序是 (w, x, y, z)，所以我们需要调整顺序为 (x, y, z, w)
            log_line = f"{timestamp} {t[0]} {t[1]} {t[2]} {q.x()} {q.y()} {q.z()} {q.w()}\n"
            
            file_handle.write(log_line)

    @staticmethod
    def log_pose_tum(file_handle, timestamp, pose):
        """
        将 gtsam.Pose3 位姿以TUM格式记录到指定的轨迹文件中。
        这是一个静态函数，用于记录快速积分过程中的位姿。

        参数:
        file_handle (File): 用于写入的文件句柄。
        timestamp (float): 时间戳（秒）
        pose (gtsam.Pose3): GTSAM位姿对象
        """
        if not file_handle or pose is None:
            return

        try:
            # 提取平移向量
            t = pose.translation()
            # 提取旋转并转换为四元数
            q = pose.rotation().toQuaternion()
            
            # 按照TUM格式写入: timestamp tx ty tz qx qy qz qw
            # gtsam.Quaternion 的顺序是 (w, x, y, z)，所以我们需要调整顺序为 (x, y, z, w)
            log_line = f"{timestamp} {t[0]} {t[1]} {t[2]} {q.x()} {q.y()} {q.z()} {q.w()}\n"
            
            file_handle.write(log_line)
        except Exception as e:
            print(f"【Debugger】: Error logging pose to trajectory file: {e}")