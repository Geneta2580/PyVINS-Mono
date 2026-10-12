"""Asynchronous metric loop verification and gravity-aligned pose-graph optimization."""

import multiprocessing as mp
import queue
import time

import cv2
import gtsam
import numpy as np


class LoopOptimization(mp.Process):
    """Own PnP and full 4DoF PGO outside the latency-sensitive VIO process."""

    def __init__(self, config, task_queue, result_queue):
        super().__init__(name="LoopOptimization", daemon=True)
        self.config = dict(config)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = mp.Event()

        self.camera_matrix = np.asarray(
            config.get('cam_intrinsics'), dtype=np.float64
        ).reshape(3, 3)
        self.T_bc = np.asarray(
            config.get('T_bc', np.eye(4).reshape(-1)), dtype=np.float64
        ).reshape(4, 4)
        self.pnp_min_points = max(4, int(config.get('loop_pnp_min_points', 12)))
        self.pnp_min_inliers = max(4, int(config.get('loop_pnp_min_inliers', 10)))
        self.pnp_min_inlier_ratio = float(
            config.get('loop_pnp_min_inlier_ratio', 0.45)
        )
        self.pnp_reprojection_error = float(
            config.get('loop_pnp_reprojection_error', 3.0)
        )
        self.pnp_confidence = float(config.get('loop_pnp_confidence', 0.999))
        self.pnp_iterations = max(1, int(config.get('loop_pnp_iterations', 200)))
        self.pnp_max_translation = float(
            config.get('loop_pnp_max_translation', 100.0)
        )
        if not 0.0 < self.pnp_min_inlier_ratio <= 1.0:
            raise ValueError("loop_pnp_min_inlier_ratio must be in (0, 1]")
        if self.pnp_reprojection_error <= 0.0:
            raise ValueError("loop_pnp_reprojection_error must be positive")
        if not 0.0 < self.pnp_confidence < 1.0:
            raise ValueError("loop_pnp_confidence must be between 0 and 1")
        if self.pnp_max_translation <= 0.0:
            raise ValueError("loop_pnp_max_translation must be positive")

        self.odom_translation_sigma = float(
            config.get('loop_pgo_odometry_translation_sigma', 0.10)
        )
        self.odom_yaw_sigma = np.deg2rad(float(
            config.get('loop_pgo_odometry_yaw_sigma_deg', 3.0)
        ))
        self.loop_translation_sigma = float(
            config.get('loop_pgo_loop_translation_sigma', 0.05)
        )
        self.loop_yaw_sigma = np.deg2rad(float(
            config.get('loop_pgo_loop_yaw_sigma_deg', 2.0)
        ))
        self.max_iterations = max(
            1, int(config.get('loop_pgo_max_iterations', 50))
        )
        self.attitude_sigma = np.deg2rad(float(
            config.get('loop_pgo_attitude_sigma_deg', 0.1)
        ))
        self.anchor_sigma = float(
            config.get('loop_pgo_anchor_sigma', 1e-6)
        )
        self.huber_k = float(config.get('loop_pgo_huber_k', 1.345))
        self.relative_error_tol = float(
            config.get('loop_pgo_relative_error_tol', 1e-5)
        )
        self.absolute_error_tol = float(
            config.get('loop_pgo_absolute_error_tol', 1e-5)
        )
        if min(
            self.odom_translation_sigma,
            self.odom_yaw_sigma,
            self.loop_translation_sigma,
            self.loop_yaw_sigma,
            self.attitude_sigma,
            self.anchor_sigma,
            self.huber_k,
            self.relative_error_tol,
            self.absolute_error_tol,
        ) <= 0.0:
            raise ValueError("all loop PGO noise and optimizer values must be positive")

        self.odometry_edges = {}
        self.loop_edges = {}

    def submit(self, job):
        if self.stop_event.is_set() or not self.is_alive():
            return False
        try:
            self.task_queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def shutdown(self):
        self.stop_event.set()
        try:
            self.task_queue.put_nowait(None)
        except queue.Full:
            pass

    def run(self):
        print("【LoopOptimization】: process started.")
        while not self.stop_event.is_set():
            try:
                job = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                result = self._process_job(job)
            except Exception as exc:
                result = {
                    'success': False,
                    'task_id': job.get('task_id'),
                    'current_frame_id': job.get('current_frame_id'),
                    'reason': f'{type(exc).__name__}: {exc}',
                }
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    pass
        print("【LoopOptimization】: process finished.")

    def _process_job(self, job):
        start = time.perf_counter()
        frame_poses = {
            int(frame_id): np.asarray(pose, dtype=np.float64).reshape(4, 4)
            for frame_id, pose in job['frame_poses'].items()
        }
        current_frame_id = int(job['current_frame_id'])
        if current_frame_id not in frame_poses:
            return self._failure(job, 'current keyframe is absent from pose graph')

        # 根据匹配关系，利用当前地图3D点，PnP计算历史KF的漂移pose
        verification = None
        for candidate in job.get('candidates', []):
            history_frame_id = int(candidate['frame_id'])
            if history_frame_id not in frame_poses:
                continue
            verification = self._solve_pnp(candidate)
            if verification is not None:
                verification['history_frame_id'] = history_frame_id
                break
        if verification is None:
            return self._failure(job, 'no loop candidate passed metric PnP')

        history_frame_id = verification['history_frame_id']
        # solvePnP returns current-camera coordinates -> history-camera coordinates.
        # 历史KF相对于当前KF的无漂移位姿
        T_history_current_camera = verification['relative_pose']
        T_history_current_body = (
            self.T_bc @ T_history_current_camera @ np.linalg.inv(self.T_bc)
        )
        loop_key = (history_frame_id, current_frame_id)
        previous_loop_edge = self.loop_edges.get(loop_key)
        self.loop_edges[loop_key] = {
            'i': history_frame_id,
            'j': current_frame_id,
            'relative_pose': T_history_current_body.copy(),
        }

        try:
            optimized_poses, optimizer_info = self._optimize_pose_graph(frame_poses)
        except Exception:
            # A failed solve must not leave an unverified constraint in the
            # process-owned persistent graph.
            if previous_loop_edge is None:
                self.loop_edges.pop(loop_key, None)
            else:
                self.loop_edges[loop_key] = previous_loop_edge
            raise
        return {
            'success': True,
            'task_id': job.get('task_id'),
            'current_frame_id': current_frame_id,
            'history_frame_id': history_frame_id,
            'input_poses': frame_poses,
            'optimized_poses': optimized_poses,
            'drift_at_submission': np.asarray(
                job['drift_at_submission'], dtype=np.float64
            ).reshape(4, 4),
            'pnp_inlier_count': verification['inlier_count'],
            'pnp_input_count': verification['input_count'],
            'pnp_inlier_ratio': verification['inlier_ratio'],
            'pnp_reprojection_rmse': verification['reprojection_rmse'],
            'relative_pose_camera': T_history_current_camera,
            'relative_pose_body': T_history_current_body,
            'optimizer_cost': optimizer_info['cost'],
            'optimizer_initial_cost': optimizer_info['initial_cost'],
            'optimizer_nfev': optimizer_info['nfev'],
            'optimizer_factor_count': optimizer_info['factor_count'],
            'elapsed_ms': (time.perf_counter() - start) * 1000.0,
        }

    @staticmethod
    def _failure(job, reason):
        return {
            'success': False,
            'task_id': job.get('task_id'),
            'current_frame_id': job.get('current_frame_id'),
            'reason': reason,
        }

    def _solve_pnp(self, candidate):
        matches = candidate.get('epipolar_matches', {})
        points_3d = np.asarray(
            matches.get('current_points_3d_camera', []), dtype=np.float64
        ).reshape(-1, 3)
        history_points = np.asarray(
            matches.get('history_undistorted_points', []), dtype=np.float64
        ).reshape(-1, 2)
        if len(points_3d) != len(history_points):
            return None
        valid = np.isfinite(points_3d).all(axis=1)
        valid &= np.isfinite(history_points).all(axis=1)
        valid &= points_3d[:, 2] > 1e-6
        points_3d = points_3d[valid]
        history_points = history_points[valid]
        if len(points_3d) < self.pnp_min_points:
            return None

        # 注意这里使用的是当前相机坐标系下的3D点（也和修正坐标系下的当前相机坐标系3D点一致）
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            history_points,
            self.camera_matrix,
            None,
            iterationsCount=self.pnp_iterations,
            reprojectionError=self.pnp_reprojection_error,
            confidence=self.pnp_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None:
            return None
        inlier_indices = inliers.reshape(-1).astype(np.int64)
        inlier_count = len(inlier_indices)
        inlier_ratio = inlier_count / float(len(points_3d))
        if (inlier_count < self.pnp_min_inliers
                or inlier_ratio < self.pnp_min_inlier_ratio):
            return None

        if inlier_count >= 6 and hasattr(cv2, 'solvePnPRefineLM'):
            rvec, tvec = cv2.solvePnPRefineLM(
                points_3d[inlier_indices],
                history_points[inlier_indices],
                self.camera_matrix,
                None,
                rvec,
                tvec,
            )
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        if (not np.isfinite(rotation).all()
                or not np.isfinite(translation).all()
                or np.linalg.norm(translation) > self.pnp_max_translation):
            return None

        projected, _ = cv2.projectPoints(
            points_3d[inlier_indices], rvec, tvec,
            self.camera_matrix, None,
        )
        residuals = projected.reshape(-1, 2) - history_points[inlier_indices]
        reprojection_rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
        relative_pose = np.eye(4, dtype=np.float64)
        relative_pose[:3, :3] = rotation
        relative_pose[:3, 3] = translation
        return {
            'relative_pose': relative_pose,
            'input_count': len(points_3d),
            'inlier_count': inlier_count,
            'inlier_ratio': inlier_ratio,
            'reprojection_rmse': reprojection_rmse,
        }

    def _register_odometry_edges(self, frame_poses):
        frame_ids = sorted(frame_poses)
        for frame_i, frame_j in zip(frame_ids[:-1], frame_ids[1:]):
            key = (frame_i, frame_j)
            if key in self.odometry_edges:
                continue
            pose_i = frame_poses[frame_i]
            pose_j = frame_poses[frame_j]
            relative = np.linalg.inv(pose_i) @ pose_j
            self.odometry_edges[key] = {
                'i': frame_i,
                'j': frame_j,
                'relative_pose': relative,
            }

    def _optimize_pose_graph(self, frame_poses):
        """Optimize Pose3 nodes while attitude factors leave only global yaw free."""
        self._register_odometry_edges(frame_poses)
        frame_ids = sorted(frame_poses)
        if len(frame_ids) < 2:
            return frame_poses, {
                'cost': 0.0,
                'initial_cost': 0.0,
                'nfev': 0,
                'factor_count': 0,
            }
        anchor_id = frame_ids[0]
        active_ids = set(frame_ids)
        odometry_edges = [
            edge for edge in self.odometry_edges.values()
            if edge['i'] in active_ids and edge['j'] in active_ids
        ]
        loop_edges = [
            edge for edge in self.loop_edges.values()
            if edge['i'] in active_ids and edge['j'] in active_ids
        ]

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        navigation_vertical = gtsam.Unit3(world_z)

        # roll和pitch噪声
        attitude_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.full(2, self.attitude_sigma, dtype=np.float64)
        )

        # 首帧固定pose
        anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.full(6, self.anchor_sigma, dtype=np.float64)
        )

        # 添加KF间的里程计SE3边噪声
        odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            self.odom_yaw_sigma,
            self.odom_yaw_sigma,
            self.odom_yaw_sigma,
            self.odom_translation_sigma,
            self.odom_translation_sigma,
            self.odom_translation_sigma,
        ], dtype=np.float64))

        # 添加KF回环边噪声
        loop_base_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            self.loop_yaw_sigma,
            self.loop_yaw_sigma,
            self.loop_yaw_sigma,
            self.loop_translation_sigma,
            self.loop_translation_sigma,
            self.loop_translation_sigma,
        ], dtype=np.float64))

        # 添加回环边鲁棒函数
        loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self.huber_k),
            loop_base_noise,
        )

        for frame_id in frame_ids:
            key = gtsam.symbol('l', frame_id)
            pose_matrix = frame_poses[frame_id]
            pose = gtsam.Pose3(pose_matrix)
            initial.insert(key, pose)
            # R_i.T * world_z is the original gravity direction in body i.
            # Keeping R_i * bRef_i == world_z fixes tilt but remains invariant
            # to a left-multiplied rotation about world Z (global yaw).
            body_vertical = gtsam.Unit3(pose_matrix[:3, :3].T @ world_z)
            graph.add(gtsam.Pose3AttitudeFactor(
                key,
                navigation_vertical,
                attitude_noise,
                body_vertical,
            ))

        anchor_key = gtsam.symbol('l', anchor_id)
        graph.add(gtsam.PriorFactorPose3(
            anchor_key, gtsam.Pose3(frame_poses[anchor_id]), anchor_noise
        ))

        # 添加KF间的里程计SE3边
        for edge in odometry_edges:
            graph.add(gtsam.BetweenFactorPose3(
                gtsam.symbol('l', edge['i']),
                gtsam.symbol('l', edge['j']),
                gtsam.Pose3(edge['relative_pose']),
                odometry_noise,
            ))

        # 添加KF回环边
        for edge in loop_edges:
            graph.add(gtsam.BetweenFactorPose3(
                gtsam.symbol('l', edge['i']),
                gtsam.symbol('l', edge['j']),
                gtsam.Pose3(edge['relative_pose']),
                loop_noise,
            ))

        initial_error = float(graph.error(initial))
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iterations)
        params.setRelativeErrorTol(self.relative_error_tol)
        params.setAbsoluteErrorTol(self.absolute_error_tol)
        params.setVerbosity('SILENT')
        params.setVerbosityLM('SILENT')
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()
        optimized_error = float(graph.error(result))

        optimized = {
            frame_id: result.atPose3(gtsam.symbol('l', frame_id)).matrix()
            for frame_id in frame_ids
        }
        if not all(np.isfinite(pose).all() for pose in optimized.values()):
            raise RuntimeError("GTSAM 4DoF pose graph produced non-finite values")
        iterations = optimizer.iterations() if hasattr(optimizer, 'iterations') else 0
        return optimized, {
            'cost': optimized_error,
            'initial_cost': initial_error,
            'nfev': int(iterations),
            'factor_count': int(graph.size()),
        }
