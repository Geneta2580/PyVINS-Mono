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
        self.params.setIntegrationCovariance(np.eye(3) * 1e-8) # 预积分协方差，通常可以设一个很小的值

        self.params.setBiasAccCovariance(np.eye(3) * accel_bias_rw_sigma**2) # 加计零偏随机游走
        self.params.setBiasOmegaCovariance(np.eye(3) * gyro_bias_rw_sigma**2) # 陀螺零偏随机游走

        # 必须设置初始化偏置协方差,GTSAM旧特性
        bias_acc_omega_init = np.eye(6) * 1e-5 
        self.params.setBiasAccOmegaInit(bias_acc_omega_init)

        self.current_bias = gtsam.imuBias.ConstantBias()
        self.max_imu_dt = config.get('max_imu_dt', 0.0125)

    @staticmethod
    def _as_sample(measurement):
        return (
            np.asarray(measurement.accel, dtype=float).reshape(3),
            np.asarray(measurement.gyro, dtype=float).reshape(3),
        )

    @staticmethod
    def _interpolate_measurement(timestamp, sample_0, sample_1):
        t0, measurement_0 = sample_0
        t1, measurement_1 = sample_1
        accel_0, gyro_0 = IMUProcessor._as_sample(measurement_0)
        accel_1, gyro_1 = IMUProcessor._as_sample(measurement_1)
        if abs(t1 - t0) < 1e-12:
            accel, gyro = accel_0, gyro_0
        else:
            ratio = (timestamp - t0) / (t1 - t0)
            accel = accel_0 + ratio * (accel_1 - accel_0)
            gyro = gyro_0 + ratio * (gyro_1 - gyro_0)

        class BoundedImuMeasurement:
            pass

        sample = BoundedImuMeasurement()
        sample.accel = accel
        sample.gyro = gyro
        return sample

    def build_bounded_imu_samples(self, timed_measurements, start_time, end_time):
        """补上 start/end 的线性插值样本，供中值积分使用。"""
        if end_time <= start_time or len(timed_measurements) < 2:
            return None
        samples = sorted(timed_measurements, key=lambda item: item[0])
        left_index = None
        right_index = None
        for index, (timestamp, _) in enumerate(samples):
            if timestamp <= start_time + 1e-9:
                left_index = index
            if right_index is None and timestamp >= end_time - 1e-9:
                right_index = index
                break
        if left_index is None or right_index is None or right_index < left_index:
            print(
                f"【IMU】: Cannot bracket [{start_time:.6f}, {end_time:.6f}] "
                f"with {len(samples)} IMU samples."
            )
            return None
        if abs(samples[left_index][0] - start_time) <= 1e-8:
            start_measurement = samples[left_index][1]
        elif left_index + 1 < len(samples):
            start_measurement = self._interpolate_measurement(
                start_time, samples[left_index], samples[left_index + 1])
        else:
            return None

        # 右边界
        if abs(samples[right_index][0] - end_time) <= 1e-8:
            end_measurement = samples[right_index][1]
        elif right_index > 0:
            end_measurement = self._interpolate_measurement(
                end_time, samples[right_index - 1], samples[right_index])
        else:
            return None

        bounded = [(start_time, start_measurement)]
        for index in range(left_index + 1, right_index):
            timestamp = samples[index][0]
            if start_time + 1e-8 < timestamp < end_time - 1e-8:
                bounded.append(samples[index])
        bounded.append((end_time, end_measurement))

        deduplicated = []
        for timestamp, measurement in bounded:
            if deduplicated and abs(timestamp - deduplicated[-1][0]) <= 1e-8:
                continue
            deduplicated.append((timestamp, measurement))
        if len(deduplicated) < 2:
            return None
        return deduplicated

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

    def pre_integration(self, measurements: List[ImuData], start_time: float, end_time: float,
                        override_bias=None, log_stats=False):
        """对已经包含起止边界的样本做中值预积分。bias 交给 GTSAM，不在这里减掉。"""
        if len(measurements) < 2:
            print("[Warning] Not enough IMU measurements to perform pre-integration.")
            return None

        current_bias = override_bias if override_bias is not None else self.current_bias
        preintegrated_measurements = gtsam.PreintegratedCombinedMeasurements(self.params, current_bias)

        dts = []
        for index in range(len(measurements) - 1):
            timestamp, measurement = measurements[index]
            next_timestamp, next_measurement = measurements[index + 1]
            dt = next_timestamp - timestamp
            if dt <= 0.0:
                print(f"【IMU】: Non-positive dt {dt:.9f} at t={timestamp:.6f}. Reject factor.")
                return None
            if dt > self.max_imu_dt:
                print(f"【IMU】: Gap warning dt={dt:.6f}s between {timestamp:.6f} and {next_timestamp:.6f}.")
            accel = 0.5 * (
                np.asarray(measurement.accel, dtype=float) + np.asarray(next_measurement.accel, dtype=float))
            gyro = 0.5 * (
                np.asarray(measurement.gyro, dtype=float) + np.asarray(next_measurement.gyro, dtype=float))
            preintegrated_measurements.integrateMeasurement(accel, gyro, dt)
            dts.append(dt)

        duration = end_time - start_time
        sum_dt = float(np.sum(dts))
        if abs(sum_dt - duration) > 1e-6:
            print(
                f"【IMU】: Integrated duration {sum_dt:.9f} differs from "
                f"frame interval {duration:.9f}. Reject factor."
            )
            return None
        delta_t = float(preintegrated_measurements.deltaTij())
        if abs(delta_t - duration) > 1e-6:
            print(f"【IMU】: PIM.deltaTij {delta_t:.9f} != interval {duration:.9f}. Reject factor.")
            return None
        if log_stats:
            print(
                f"【IMU】: samples={len(measurements)} dt[min/mean/max]="
                f"{min(dts):.6f}/{np.mean(dts):.6f}/{max(dts):.6f} "
                f"bounds=[{measurements[0][0]:.6f}, {measurements[-1][0]:.6f}] "
                f"deltaTij={delta_t:.6f} interval={duration:.6f}"
            )
        return preintegrated_measurements