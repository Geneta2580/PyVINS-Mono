import numpy as np
import gtsam
from collections import deque
from typing import List, Tuple

ImuData = Tuple[float, any]

class IMUProcessor:
    def __init__(self, config):
        self.g = config.get('gravity', 9.81)
    
        # 从config文件获取IMU参数
        accel_noise_sigma = config.get('accel_noise_sigma', 1e-2)
        gyro_noise_sigma = config.get('gyro_noise_sigma', 1e-3)
        accel_bias_rw_sigma = config.get('accel_bias_rw_sigma', 1e-4)
        gyro_bias_rw_sigma = config.get('gyro_bias_rw_sigma', 1e-5)
        
        # 传递到GTSAM参数
        self.params = gtsam.PreintegrationCombinedParams.MakeSharedU(self.g) # 重力补偿参数

        self.params.setAccelerometerCovariance(np.eye(3) * accel_noise_sigma**2) # 加计协方差
        self.params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2) # 陀螺协方差
        self.params.setIntegrationCovariance(np.eye(3) * 1e-6) # 预积分协方差，通常可以设一个很小的值

        self.params.setBiasAccCovariance(np.eye(3) * accel_bias_rw_sigma**2) # 加计零偏随机游走
        self.params.setBiasOmegaCovariance(np.eye(3) * gyro_bias_rw_sigma**2) # 陀螺零偏随机游走

        self.current_bias = gtsam.imuBias.ConstantBias()

    @staticmethod
    def get_imu_interval_with(imu_buffer_deque: deque, end_time: float) -> Tuple[List[ImuData], deque]:
        measurements_to_process = []

        while len(imu_buffer_deque) > 0 and imu_buffer_deque[0][0] <= end_time:
            # 添加到IMU测量列表，同时从缓冲区中删除这些数据
            measurements_to_process.append(imu_buffer_deque.popleft()) 

        return measurements_to_process, imu_buffer_deque

    def update_bias(self, new_bias):
        self.current_bias = new_bias

    def fast_integration(self, dt, latest_nav_state, current_imu_data):
        # 从 self.current_bias 中提取偏置值，每次优化后都会更新
        accel_bias = np.array(self.current_bias.accelerometer())
        gyro_bias = np.array(self.current_bias.gyroscope())
        
        latest_pose = latest_nav_state['pose']
        latest_velocity = latest_nav_state['velocity']

        # 转换为 numpy 数组
        latest_vel = np.array(latest_velocity) if not isinstance(latest_velocity, np.ndarray) else latest_velocity
        accel_meas = np.array(current_imu_data.accel)
        gyro_meas = np.array(current_imu_data.gyro)
        
        # 获取旋转矩阵
        latest_rotation = latest_pose.rotation()  # gtsam.Rot3
        
        # 中值积分
        # 第一步：使用当前偏置补偿 IMU 测量值
        un_acc0 = latest_rotation.matrix() @ (accel_meas - accel_bias) - np.array([0, 0, self.g])
        un_gyr = gyro_meas - gyro_bias  # 角速度（已补偿偏置）
        
        # 计算旋转增量（使用 GTSAM 的指数映射）
        delta_rotation = gtsam.Rot3.Expmap(un_gyr * dt)
        mid_rotation = latest_rotation.compose(delta_rotation)
        
        # 第二步：使用中间旋转计算加速度
        un_acc1 = mid_rotation.matrix() @ (accel_meas - accel_bias) - np.array([0, 0, self.g])
        un_acc = 0.5 * (un_acc0 + un_acc1)
        
        # 进行快速积分
        # 更新位置
        latest_position = np.array(latest_pose.translation())
        current_position = latest_position + dt * latest_vel + 0.5 * dt * dt * un_acc
        
        # 更新速度
        current_velocity = latest_vel + dt * un_acc
        
        # 构建新的位姿
        current_pose_np = np.eye(4)
        current_pose_np[:3, :3] = mid_rotation.matrix()
        current_pose_np[:3, 3] = current_position
        current_pose = gtsam.Pose3(current_pose_np)
        
        return current_pose, current_velocity

    def pre_integration(self, measurements: List[ImuData], start_time: float, end_time: float, override_bias = None):

        if len(measurements) < 2:
            print("[Warning] Not enough IMU measurements to perform pre-integration.")
            return None

        if override_bias is not None:
            # print("【IMU_process】: Using override bias")
            current_bias = override_bias
        else:
            # 假设每次都从零偏置开始，在实际系统中，这里应该传入上一个关键帧优化后的偏置
            current_bias = self.current_bias

        preintegrated_measurements = gtsam.PreintegratedCombinedMeasurements(self.params, current_bias)

        # 逐段积分IMU数据
        last_timestamp = start_time
        for imu_data in measurements:
            timestamp, data = imu_data

            if timestamp >= last_timestamp:
                dt = timestamp - last_timestamp # 注意这里的单位应该是s
                if dt <=0:
                    continue

                preintegrated_measurements.integrateMeasurement(data.accel, data.gyro, dt)

                last_timestamp = timestamp

        # 对最后一段IMU进行积分
        final_dt = end_time - last_timestamp
        if final_dt > 0:
            last_accel = measurements[-1][1].accel
            last_gyro = measurements[-1][1].gyro
            preintegrated_measurements.integrateMeasurement(last_accel, last_gyro, final_dt)

        return preintegrated_measurements