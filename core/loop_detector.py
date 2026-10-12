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
        self.display_width = max(
            160, int(config.get('loop_detector_visualization_width', 480))
        )
        self.display_jpeg_quality = int(
            np.clip(config.get('loop_detector_visualization_jpeg_quality', 80), 1, 100)
        )
        self.submitted_count = 0
        self.dropped_count = 0
        self.processed_count = 0

    def submit_keyframe(self, frame):
        """Queue a keyframe without waiting; discard the oldest task if full."""
        if self.stop_event.is_set() or not self.is_alive():
            return False

        # 获取关键帧图像
        image = frame.get_image()
        if image is None:
            return False
            
        # 创建回环检测任务
        task = LoopDetectionTask(
            frame_id=int(frame.get_id()),
            timestamp=float(frame.get_timestamp()),
            image=image,
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

        # 查找候选帧
        candidates = self._find_candidates(task.frame_id, descriptor)
        current_image_jpeg = self._encode_display_image(task.image)
        best_image_jpeg = None
        if candidates:
            best_database_index = candidates[0]['_database_index']
            best_image_jpeg = self._display_images[best_database_index]

        # 数据库索引仅供回环线程内部取图，不能暴露给 estimator。
        for candidate in candidates:
            candidate.pop('_database_index', None)

        # 更新数据库
        self._append_descriptor(descriptor)
        self._frame_ids.append(task.frame_id)
        self._timestamps.append(task.timestamp)
        self._display_images.append(current_image_jpeg)

        # 返回结果
        return {
            'frame_id': task.frame_id,
            'timestamp': task.timestamp,
            'descriptor': descriptor,
            'candidates': candidates,
            'current_image_jpeg': current_image_jpeg,
            'best_image_jpeg': best_image_jpeg,
            'inference_ms': (time.perf_counter() - start_time) * 1000.0,
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
        order = np.argsort(scores)[::-1][:self.top_k]

        # 提取得分最高的K个候选帧
        candidates = []
        for local_index in order:
            score = float(scores[local_index])
            if score < self.similarity_threshold:
                continue
            database_index = int(local_index)
            candidates.append({
                'frame_id': self._frame_ids[database_index],
                'timestamp': self._timestamps[database_index],
                'similarity': score,
                '_database_index': database_index,
            })
        return candidates

    def _publish_result(self, result):
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            # Results are advisory until loop optimization is implemented.
            # Prefer the newest result and never back-pressure the GPU worker.
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                pass
