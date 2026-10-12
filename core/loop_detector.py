"""Asynchronous SALAD-based loop-place recognition.

The worker owns the model and descriptor database.  It never mutates the VIO
map; the estimator is the sole consumer of the lightweight detection results.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from bisect import bisect_right
import importlib.util
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class LoopDetectionTask:
    frame_id: int
    timestamp: float
    image: np.ndarray
    tracked_raw_points: np.ndarray
    tracked_undistorted_points: np.ndarray
    tracked_feature_ids: np.ndarray
    tracked_points_3d_camera: np.ndarray


@dataclass
class LoopKeyframeData:
    """Long-lived loop data, independent of the estimator's sliding window."""

    frame_id: int
    timestamp: float
    image_jpeg: bytes
    image_shape: tuple
    global_descriptor: np.ndarray
    tracked_raw_points: np.ndarray
    tracked_undistorted_points: np.ndarray
    tracked_feature_ids: np.ndarray
    tracked_points_3d_camera: np.ndarray
    tracked_brief_descriptors: np.ndarray
    fast_raw_points: np.ndarray
    fast_undistorted_points: np.ndarray
    fast_brief_descriptors: np.ndarray


class LoopDetector(threading.Thread):
    """Non-blocking keyframe consumer which computes SALAD descriptors."""

    def __init__(self, config, result_queue):
        super().__init__(name="LoopDetector", daemon=True)
        self.config = config
        self.result_queue = result_queue
        self.task_queue = queue.Queue(
            maxsize=max(1, int(config.get('loop_detector_queue_size', 2)))
        )
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.model = None
        self.torch = None
        self.device = None
        self.last_error = None

        image_size = config.get('loop_detector_image_size', [322, 322])
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        if len(image_size) != 2:
            raise ValueError("loop_detector_image_size must contain [height, width]")
        self.image_size = tuple(int(value) for value in image_size)
        if any(value <= 0 or value % 14 != 0 for value in self.image_size):
            raise ValueError("SALAD input height and width must be positive multiples of 14")

        self.top_k = max(1, int(config.get('loop_detector_top_k', 5)))
        self.similarity_threshold = float(
            config.get('loop_detector_similarity_threshold', 0.85)
        )
        self.min_frame_gap = max(0, int(config.get('loop_detector_min_frame_gap', 30)))
        self.use_amp = bool(config.get('loop_detector_use_amp', True))
        self._descriptor_matrix = None
        self._descriptor_count = 0
        self._frame_ids = []
        self._timestamps = []
        self._display_images = []
        self.keyframe_database = {}
        self._archive_lock = threading.Lock()
        self._archived_frame_ids = set()
        self.display_width = max(
            160, int(config.get('loop_detector_visualization_width', 480))
        )
        self.display_jpeg_quality = int(
            np.clip(config.get('loop_detector_visualization_jpeg_quality', 80), 1, 100)
        )
        # FAST响应阈值：越大只保留越明显的角点，角点数量通常越少。
        fast_threshold = max(
            1, int(config.get('loop_detector_fast_threshold', 20))
        )
        # FAST角点数量上限：限制每个历史关键帧用于BRIEF匹配的候选点数。
        self.fast_max_keypoints = max(
            1, int(config.get('loop_detector_fast_max_keypoints', 500))
        )
        # Lowe比率阈值：要求最佳距离小于次优距离乘该值，越小筛选越严格。
        self.brief_ratio_threshold = float(
            config.get('loop_detector_brief_ratio_threshold', 0.80)
        )
        # BRIEF最大汉明距离：超过该绝对距离的匹配会被拒绝，越小越严格。
        self.brief_max_distance = max(
            0, int(config.get('loop_detector_brief_max_distance', 80))
        )
        # 回环极线RANSAC阈值，单位为去畸变虚拟针孔图像上的像素。
        self.epipolar_ransac_threshold = float(
            config.get('loop_detector_ransac_threshold', 1.0)
        )
        if self.epipolar_ransac_threshold <= 0.0:
            raise ValueError("loop_detector_ransac_threshold must be positive")
        # RANSAC找到无外点模型的置信度。
        self.epipolar_ransac_prob = float(
            config.get('loop_detector_ransac_prob', 0.999)
        )
        if not 0.0 < self.epipolar_ransac_prob < 1.0:
            raise ValueError("loop_detector_ransac_prob must be between 0 and 1")
        # 少于该数量时不启动本质矩阵几何验证。
        self.epipolar_ransac_min_matches = max(
            5, int(config.get('loop_detector_ransac_min_matches', 8))
        )
        self.fast_detector = cv2.FastFeatureDetector_create(
            threshold=fast_threshold,
            nonmaxSuppression=True,
        )
        if not hasattr(cv2, 'xfeatures2d'):
            raise RuntimeError(
                "OpenCV contrib is required for BRIEF; install "
                "opencv-contrib-python instead of opencv-python"
            )
        # BRIEF描述子字节数：支持16/32/64字节，越长区分力和计算开销通常越高。
        brief_bytes = int(config.get('loop_detector_brief_bytes', 32))
        if brief_bytes not in (16, 32, 64):
            raise ValueError("loop_detector_brief_bytes must be 16, 32, or 64")
        self.brief_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
            brief_bytes, False
        )
        self.brief_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        cam_intrinsics_raw = config.get('cam_intrinsics', np.eye(3).reshape(-1).tolist())
        self.cam_intrinsics = np.asarray(cam_intrinsics_raw, dtype=np.float64).reshape(3, 3)
        self.use_undistort = bool(config.get('use_undistort', True))
        self.distortion_model = str(config.get('distortion_model', 'radtan')).lower()
        self.distortion_coefficients = np.asarray(
            config.get('distortion_coefficients', []), dtype=np.float64
        ).reshape(-1)
        self.submitted_count = 0
        self.dropped_count = 0
        self.processed_count = 0

    def submit_keyframe(self, frame, tracked_points_3d_camera=None):
        """Queue a keyframe without waiting; discard the oldest task if full."""
        if self.stop_event.is_set() or not self.is_alive():
            return False

        # 获取关键帧图像
        image = frame.get_image()
        if image is None:
            return False
        tracked_raw_points = frame.get_tracked_raw_features()
        tracked_undistorted_points = frame.get_tracked_undistorted_features()
        tracked_feature_ids = frame.get_tracked_feature_ids()
        if (tracked_raw_points is None or tracked_undistorted_points is None
                or tracked_feature_ids is None):
            return False
        tracked_count = len(np.asarray(tracked_feature_ids).reshape(-1))
        if tracked_points_3d_camera is None:
            tracked_points_3d_camera = np.full(
                (tracked_count, 3), np.nan, dtype=np.float64
            )
        tracked_points_3d_camera = np.asarray(
            tracked_points_3d_camera, dtype=np.float64
        ).reshape(-1, 3)
        if len(tracked_points_3d_camera) != tracked_count:
            raise ValueError("tracked 3D points and feature IDs are not aligned")
            
        # 创建回环检测任务
        task = LoopDetectionTask(
            frame_id=int(frame.get_id()),
            timestamp=float(frame.get_timestamp()),
            image=image,
            tracked_raw_points=np.asarray(
                tracked_raw_points, dtype=np.float32
            ).reshape(-1, 2).copy(),
            tracked_undistorted_points=np.asarray(
                tracked_undistorted_points, dtype=np.float32
            ).reshape(-1, 2).copy(),
            tracked_feature_ids=np.asarray(
                tracked_feature_ids, dtype=np.int64
            ).reshape(-1).copy(),
            tracked_points_3d_camera=tracked_points_3d_camera.copy(),
        )
        try:
            self.task_queue.put_nowait(task)
        except queue.Full:
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
                self.dropped_count += 1
            except queue.Empty:
                pass
            try:
                self.task_queue.put_nowait(task)
            except queue.Full:
                self.dropped_count += 1
                return False
        self.submitted_count += 1
        return True

    def shutdown(self):
        self.stop_event.set()
        # Wake a worker waiting on an empty queue without ever blocking VIO.
        try:
            self.task_queue.put_nowait(None)
        except queue.Full:
            pass

    def mark_frame_archived(self, frame_id):
        """Allow a marginalized keyframe to participate as loop history."""
        with self._archive_lock:
            self._archived_frame_ids.add(int(frame_id))

    def run(self):
        print("【LoopDetector】: thread started; loading SALAD in the worker.")
        try:
            # 加载模型（只执行一次）
            self._load_model()
            self.ready_event.set()
            print(f"【LoopDetector】: SALAD ready on {self.device}.")
            while not self.stop_event.is_set():
                try:
                    task = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if task is None:
                    self.task_queue.task_done()
                    break
                try:
                    # 处理回环检测任务
                    result = self._process_task(task)
                    # 发布回环检测结果
                    self._publish_result(result)
                    self.processed_count += 1
                    
                except Exception as exc:
                    self.last_error = exc
                    print(
                        f"【LoopDetector】: inference failed for frame "
                        f"{task.frame_id}: {exc}"
                    )
                finally:
                    self.task_queue.task_done()
        except Exception as exc:
            self.last_error = exc
            print(f"【LoopDetector】: disabled because SALAD failed to load: {exc}")
        finally:
            self.stop_event.set()
            print(
                f"【LoopDetector】: thread finished; submitted={self.submitted_count} "
                f"processed={self.processed_count} dropped={self.dropped_count}."
            )

    def _resolve_salad_repository(self):
        configured = Path(
            self.config.get('loop_detector_salad_path', 'submodule/salad')
        ).expanduser()
        if not configured.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            configured = project_root / configured
        repository = configured.resolve()
        if not (repository / 'hubconf.py').is_file():
            raise FileNotFoundError(
                f"SALAD submodule is unavailable at {repository}. Run "
                "'git submodule update --init --recursive'."
            )
        return repository

    def _resolve_local_path(self, config_key, default):
        configured = Path(self.config.get(config_key, default)).expanduser()
        if not configured.is_absolute():
            configured = Path(__file__).resolve().parents[1] / configured
        return configured.resolve()

    def _load_salad_aggregator_class(self, repository):
        module_path = repository / 'models' / 'aggregators' / 'salad.py'
        spec = importlib.util.spec_from_file_location(
            '_pyvins_salad_aggregator', module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load SALAD aggregator from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SALAD

    def _load_model(self):
        # Imports stay inside the worker so disabled loop detection adds no AI
        # dependency or import latency to the VIO threads.
        import torch

        requested_device = str(self.config.get('loop_detector_device', 'cuda'))
        if requested_device == 'auto':
            requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested_device.startswith('cuda') and not torch.cuda.is_available():
            if not bool(self.config.get('loop_detector_allow_cpu', False)):
                raise RuntimeError(
                    "CUDA was requested but is unavailable; set "
                    "loop_detector_allow_cpu: true only for debugging"
                )
            requested_device = 'cpu'

        self.torch = torch
        self.device = torch.device(requested_device)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        salad_repository = self._resolve_salad_repository()
        dinov2_repository = self._resolve_local_path(
            'loop_detector_dinov2_path', 'submodule/dinov2'
        )
        salad_checkpoint = self._resolve_local_path(
            'loop_detector_salad_checkpoint', 'checkpoints/dino_salad.ckpt'
        )
        dinov2_checkpoint = self._resolve_local_path(
            'loop_detector_dinov2_checkpoint',
            'checkpoints/dinov2_vitb14_pretrain.pth',
        )
        for required in (
                dinov2_repository / 'hubconf.py', salad_checkpoint,
                dinov2_checkpoint):
            if not required.is_file():
                raise FileNotFoundError(f"required local model asset is missing: {required}")

        # Load both code and weights from explicit local paths.  No model
        # download is allowed in the runtime inference path.
        dinov2 = torch.hub.load(
            str(dinov2_repository),
            'dinov2_vitb14',
            source='local',
            pretrained=False,
        )
        dinov2_state_dict = torch.load(
            dinov2_checkpoint, map_location='cpu', weights_only=True
        )
        dinov2.load_state_dict(dinov2_state_dict, strict=True)
        SALAD = self._load_salad_aggregator_class(salad_repository)

        class DINOv2Backbone(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.num_trainable_blocks = 4

            def forward(self, image):
                batch, _, height, width = image.shape
                tokens = self.model.prepare_tokens_with_masks(image)
                with torch.no_grad():
                    for block in self.model.blocks[:-self.num_trainable_blocks]:
                        tokens = block(tokens)
                tokens = tokens.detach()
                for block in self.model.blocks[-self.num_trainable_blocks:]:
                    tokens = block(tokens)
                tokens = self.model.norm(tokens)
                global_token = tokens[:, 0]
                features = tokens[:, 1:].reshape(
                    batch, height // 14, width // 14, 768
                ).permute(0, 3, 1, 2)
                return features, global_token

        class SALADInferenceModel(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = DINOv2Backbone(backbone)
                self.aggregator = SALAD(
                    num_channels=768,
                    num_clusters=64,
                    cluster_dim=128,
                    token_dim=256,
                )

            def forward(self, image):
                return self.aggregator(self.backbone(image))

        self.model = SALADInferenceModel(dinov2)
        state_dict = torch.load(
            salad_checkpoint, map_location='cpu', weights_only=True
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.eval().to(self.device)

    def _preprocess(self, image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"unsupported keyframe image shape {image.shape}")

        height, width = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        image = np.ascontiguousarray(image)
        tensor = self.torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = tensor.sub_(mean).div_(std).unsqueeze(0)
        return tensor.to(self.device, non_blocking=self.device.type == 'cuda')

    def _process_task(self, task):
        start_time = time.perf_counter()

        # 预处理图像
        tensor = self._preprocess(task.image)

        # 使用AMP
        amp_context = nullcontext()
        if self.device.type == 'cuda' and self.use_amp:
            amp_context = self.torch.autocast(device_type='cuda', dtype=self.torch.float16)

        # 推理
        with self.torch.inference_mode(), amp_context:
            descriptor = self.model(tensor)

        # 后处理
        descriptor = (
            descriptor.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        )
        norm = float(np.linalg.norm(descriptor))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("SALAD produced an invalid descriptor")
        descriptor /= norm

        current_image_jpeg = self._encode_display_image(task.image)
        current_keyframe = self._build_loop_keyframe_data(
            task, descriptor, current_image_jpeg
        )

        # SALAD查找候选帧，再对每个候选执行KLT-BRIEF/FAST-BRIEF匹配。
        candidates = self._find_candidates(task.frame_id, descriptor)
        best_image_jpeg = None
        for candidate in candidates:
            history = self.keyframe_database.get(candidate['frame_id'])
            if history is None:
                candidate['brief_matches'] = self._empty_brief_matches()
            else:
                candidate['brief_matches'] = self._match_brief(
                    current_keyframe, history
                )
            candidate['brief_match_count'] = int(
                len(candidate['brief_matches']['distances'])
            )
            candidate['epipolar_matches'] = self._epipolar_ransac(
                candidate['brief_matches']
            )
            candidate['epipolar_inlier_count'] = int(
                len(candidate['epipolar_matches']['distances'])
            )
            candidate['epipolar_inlier_ratio'] = (
                candidate['epipolar_inlier_count']
                / candidate['brief_match_count']
                if candidate['brief_match_count'] > 0 else 0.0
            )
        if candidates:
            best_keyframe = self.keyframe_database[candidates[0]['frame_id']]
            best_image_jpeg = best_keyframe.image_jpeg
            best_image_shape = best_keyframe.image_shape
        else:
            best_image_shape = None

        # 数据库索引仅供回环线程内部取图，不能暴露给 estimator。
        for candidate in candidates:
            candidate.pop('_database_index', None)

        # 更新数据库
        self._append_descriptor(descriptor)
        self._frame_ids.append(task.frame_id)
        self._timestamps.append(task.timestamp)
        self._display_images.append(current_image_jpeg)
        self.keyframe_database[task.frame_id] = current_keyframe

        # 返回结果
        return {
            'frame_id': task.frame_id,
            'timestamp': task.timestamp,
            'descriptor': descriptor,
            'candidates': candidates,
            'current_image_jpeg': current_image_jpeg,
            'current_image_shape': current_keyframe.image_shape,
            'best_image_jpeg': best_image_jpeg,
            'best_image_shape': best_image_shape,
            'inference_ms': (time.perf_counter() - start_time) * 1000.0,
        }

    @staticmethod
    def _as_gray(image):
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"unsupported keyframe image shape {image.shape}")

    def _undistort_local_points(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if len(points) == 0 or not self.use_undistort:
            return points.reshape(-1, 2).astype(np.float32)
        if self.distortion_model in ('fisheye', 'equidistant'):
            corrected = cv2.fisheye.undistortPoints(
                points,
                self.cam_intrinsics,
                self.distortion_coefficients.reshape(4, 1),
                R=np.eye(3, dtype=np.float64),
                P=self.cam_intrinsics,
            )
        else:
            corrected = cv2.undistortPoints(
                points,
                self.cam_intrinsics,
                self.distortion_coefficients,
                P=self.cam_intrinsics,
            )
        return corrected.reshape(-1, 2).astype(np.float32)

    def _compute_tracked_brief(self, gray, raw_points, undistorted_points,
                               feature_ids, points_3d_camera):
        raw_points = np.asarray(raw_points, dtype=np.float32).reshape(-1, 2)
        undistorted_points = np.asarray(
            undistorted_points, dtype=np.float32
        ).reshape(-1, 2)
        feature_ids = np.asarray(feature_ids, dtype=np.int64).reshape(-1)
        points_3d_camera = np.asarray(
            points_3d_camera, dtype=np.float64
        ).reshape(-1, 3)
        if not (len(raw_points) == len(undistorted_points) == len(feature_ids)
                == len(points_3d_camera)):
            raise ValueError("tracked 2D/3D points and feature IDs are not aligned")

        height, width = gray.shape[:2]
        keypoints = []
        for index, point in enumerate(raw_points):
            x, y = float(point[0]), float(point[1])
            if not np.isfinite(x + y) or x < 0 or y < 0 or x >= width or y >= height:
                continue
            keypoint = cv2.KeyPoint(x, y, 31.0)
            keypoint.class_id = index
            keypoints.append(keypoint)
        if not keypoints:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, self.brief_extractor.descriptorSize()), dtype=np.uint8),
            )

        retained_keypoints, descriptors = self.brief_extractor.compute(gray, keypoints)
        if descriptors is None or not retained_keypoints:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, self.brief_extractor.descriptorSize()), dtype=np.uint8),
            )
        retained_indices = np.asarray(
            [keypoint.class_id for keypoint in retained_keypoints], dtype=np.int64
        )
        if np.any(retained_indices < 0) or np.any(retained_indices >= len(raw_points)):
            raise RuntimeError("BRIEF did not preserve tracked keypoint indices")
        return (
            raw_points[retained_indices],
            undistorted_points[retained_indices],
            feature_ids[retained_indices],
            points_3d_camera[retained_indices],
            np.asarray(descriptors, dtype=np.uint8),
        )

    def _compute_fast_brief(self, gray):
        keypoints = self.fast_detector.detect(gray, None)
        keypoints = sorted(
            keypoints, key=lambda keypoint: keypoint.response, reverse=True
        )[:self.fast_max_keypoints]
        if not keypoints:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, self.brief_extractor.descriptorSize()), dtype=np.uint8),
            )
        keypoints, descriptors = self.brief_extractor.compute(gray, keypoints)
        if descriptors is None or not keypoints:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, self.brief_extractor.descriptorSize()), dtype=np.uint8),
            )
        raw_points = np.asarray(
            [keypoint.pt for keypoint in keypoints], dtype=np.float32
        )
        return (
            raw_points,
            self._undistort_local_points(raw_points),
            np.asarray(descriptors, dtype=np.uint8),
        )

    def _build_loop_keyframe_data(self, task, descriptor, image_jpeg):
        gray = self._as_gray(task.image)
        (tracked_raw, tracked_undistorted, tracked_ids, tracked_points_3d,
         tracked_descriptors) = self._compute_tracked_brief(
            gray,
            task.tracked_raw_points,
            task.tracked_undistorted_points,
            task.tracked_feature_ids,
            task.tracked_points_3d_camera,
        )
        fast_raw, fast_undistorted, fast_descriptors = (
            self._compute_fast_brief(gray)
        )
        return LoopKeyframeData(
            frame_id=task.frame_id,
            timestamp=task.timestamp,
            image_jpeg=image_jpeg,
            image_shape=tuple(task.image.shape[:2]),
            global_descriptor=descriptor,
            tracked_raw_points=tracked_raw,
            tracked_undistorted_points=tracked_undistorted,
            tracked_feature_ids=tracked_ids,
            tracked_points_3d_camera=tracked_points_3d,
            tracked_brief_descriptors=tracked_descriptors,
            fast_raw_points=fast_raw,
            fast_undistorted_points=fast_undistorted,
            fast_brief_descriptors=fast_descriptors,
        )

    @staticmethod
    def _empty_brief_matches():
        return {
            'current_raw_points': np.empty((0, 2), dtype=np.float32),
            'history_raw_points': np.empty((0, 2), dtype=np.float32),
            'current_undistorted_points': np.empty((0, 2), dtype=np.float32),
            'history_undistorted_points': np.empty((0, 2), dtype=np.float32),
            'current_feature_ids': np.empty((0,), dtype=np.int64),
            'current_points_3d_camera': np.empty((0, 3), dtype=np.float64),
            'distances': np.empty((0,), dtype=np.float32),
        }

    def _match_brief(self, current, history):
        current_descriptors = current.tracked_brief_descriptors
        history_descriptors = history.fast_brief_descriptors
        if len(current_descriptors) == 0 or len(history_descriptors) < 2:
            return self._empty_brief_matches()

        forward_pairs = self.brief_matcher.knnMatch(
            current_descriptors, history_descriptors, k=2
        )
        reverse_matches = self.brief_matcher.match(
            history_descriptors, current_descriptors
        )
        reverse_best = {
            match.queryIdx: match.trainIdx for match in reverse_matches
        }
        accepted = []
        for pair in forward_pairs:
            if len(pair) != 2:
                continue
            best, second = pair
            if best.distance > self.brief_max_distance:
                continue
            if best.distance >= self.brief_ratio_threshold * second.distance:
                continue
            if reverse_best.get(best.trainIdx) != best.queryIdx:
                continue
            accepted.append(best)
        accepted.sort(key=lambda match: match.distance)
        if not accepted:
            return self._empty_brief_matches()

        current_indices = np.asarray(
            [match.queryIdx for match in accepted], dtype=np.int64
        )
        history_indices = np.asarray(
            [match.trainIdx for match in accepted], dtype=np.int64
        )
        return {
            'current_raw_points': current.tracked_raw_points[current_indices],
            'history_raw_points': history.fast_raw_points[history_indices],
            'current_undistorted_points': (
                current.tracked_undistorted_points[current_indices]
            ),
            'history_undistorted_points': (
                history.fast_undistorted_points[history_indices]
            ),
            'current_feature_ids': current.tracked_feature_ids[current_indices],
            'current_points_3d_camera': (
                current.tracked_points_3d_camera[current_indices]
            ),
            'distances': np.asarray(
                [match.distance for match in accepted], dtype=np.float32
            ),
        }

    def _epipolar_ransac(self, matches):
        """Keep correspondences consistent with one calibrated two-view model."""
        match_count = len(matches['distances'])
        if match_count < self.epipolar_ransac_min_matches:
            return self._empty_brief_matches()

        current_points = np.asarray(
            matches['current_undistorted_points'], dtype=np.float64
        ).reshape(-1, 2)
        history_points = np.asarray(
            matches['history_undistorted_points'], dtype=np.float64
        ).reshape(-1, 2)
        if len(current_points) != match_count or len(history_points) != match_count:
            raise ValueError("BRIEF match arrays are not aligned for epipolar RANSAC")

        finite_mask = np.isfinite(current_points).all(axis=1)
        finite_mask &= np.isfinite(history_points).all(axis=1)
        finite_indices = np.flatnonzero(finite_mask)
        if len(finite_indices) < self.epipolar_ransac_min_matches:
            return self._empty_brief_matches()

        try:
            essential_matrix, ransac_mask = cv2.findEssentialMat(
                current_points[finite_indices],
                history_points[finite_indices],
                cameraMatrix=self.cam_intrinsics,
                method=cv2.RANSAC,
                prob=self.epipolar_ransac_prob,
                threshold=self.epipolar_ransac_threshold,
            )
        except cv2.error as exc:
            print(f"【LoopDetector】: epipolar RANSAC failed: {exc}")
            return self._empty_brief_matches()
        if essential_matrix is None or ransac_mask is None:
            return self._empty_brief_matches()

        local_inlier_mask = ransac_mask.reshape(-1).astype(bool)
        if len(local_inlier_mask) != len(finite_indices):
            return self._empty_brief_matches()
        inlier_indices = finite_indices[local_inlier_mask]
        return {
            'current_raw_points': matches['current_raw_points'][inlier_indices],
            'history_raw_points': matches['history_raw_points'][inlier_indices],
            'current_undistorted_points': (
                matches['current_undistorted_points'][inlier_indices]
            ),
            'history_undistorted_points': (
                matches['history_undistorted_points'][inlier_indices]
            ),
            'current_feature_ids': matches['current_feature_ids'][inlier_indices],
            'current_points_3d_camera': (
                matches['current_points_3d_camera'][inlier_indices]
            ),
            'distances': matches['distances'][inlier_indices],
        }

    def _encode_display_image(self, image):
        """Store a compact keyframe thumbnail for loop-result visualization."""
        height, width = image.shape[:2]
        if width > self.display_width:
            scale = self.display_width / float(width)
            display = cv2.resize(
                image,
                (self.display_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            display = image
        success, encoded = cv2.imencode(
            '.jpg', display,
            [cv2.IMWRITE_JPEG_QUALITY, self.display_jpeg_quality],
        )
        if not success:
            raise RuntimeError("failed to encode loop visualization image")
        return encoded.tobytes()

    def _append_descriptor(self, descriptor):
        """Append without rebuilding the complete descriptor database."""
        descriptor = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        if self._descriptor_matrix is None:
            self._descriptor_matrix = np.empty(
                (256, descriptor.size), dtype=np.float32
            )
        elif descriptor.size != self._descriptor_matrix.shape[1]:
            raise ValueError(
                f"descriptor dimension changed from "
                f"{self._descriptor_matrix.shape[1]} to {descriptor.size}"
            )
        elif self._descriptor_count == self._descriptor_matrix.shape[0]:
            expanded = np.empty(
                (self._descriptor_matrix.shape[0] * 2,
                 self._descriptor_matrix.shape[1]),
                dtype=np.float32,
            )
            expanded[:self._descriptor_count] = self._descriptor_matrix
            self._descriptor_matrix = expanded
        self._descriptor_matrix[self._descriptor_count] = descriptor
        self._descriptor_count += 1

    def _find_candidates(self, frame_id, descriptor):

        # 时间上不能过于接近
        eligible_count = bisect_right(
            self._frame_ids, frame_id - self.min_frame_gap
        )
        if eligible_count == 0:
            return []

        # 计算描述子相似度得分
        database = self._descriptor_matrix[:eligible_count]
        scores = database @ descriptor
        order = np.argsort(scores)[::-1]
        with self._archive_lock:
            archived_frame_ids = self._archived_frame_ids.copy()

        # 提取得分最高的K个候选帧
        candidates = []
        for local_index in order:
            score = float(scores[local_index])
            if score < self.similarity_threshold:
                break
            database_index = int(local_index)
            if self._frame_ids[database_index] not in archived_frame_ids:
                continue
            candidates.append({
                'frame_id': self._frame_ids[database_index],
                'timestamp': self._timestamps[database_index],
                'similarity': score,
                '_database_index': database_index,
            })
            if len(candidates) >= self.top_k:
                break
        return candidates

    def _publish_result(self, result):
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            # Prefer the newest result and never back-pressure the GPU worker;
            # Estimator snapshots the map only when it can submit optimization.
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                pass
