import open3d as o3d
import numpy as np
import threading
import queue

class Viewer3D(threading.Thread):
    def __init__(self, viewer_queue):
        super().__init__(daemon=True)
        self.viewer_queue = viewer_queue
        self.is_running = False
        self.lock = threading.Lock()

        self.landmarks = {}
        self.poses = {}
        self.global_landmarks = {}
        self.global_poses = {}

        self.vis = None
        self.local_cloud = o3d.geometry.PointCloud()
        self.global_cloud = o3d.geometry.PointCloud()
        self.local_cameras = {}
        self.global_cameras = {}
        self.scene_initialized = False

    def run(self):
        self.is_running = True

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("PyVINS-Fusion Viewer")
        self.vis.add_geometry(self.global_cloud)
        self.vis.add_geometry(self.local_cloud)

        print("【Viewer】 thread started. Waiting for data...")

        while self.is_running:
            try:
                data = self.viewer_queue.get(timeout=0.01)
                if data is None:
                    break
                with self.lock:
                    if 'landmarks' in data:
                        self.landmarks = data['landmarks']
                    if 'poses' in data:
                        self.poses = data['poses']
                    if 'global_landmarks' in data:
                        self.global_landmarks = data['global_landmarks']
                    if 'global_poses' in data:
                        self.global_poses = data['global_poses']
            except queue.Empty:
                pass

            self._render_current_scene()

            if not self.vis.poll_events():
                break

        self.is_running = False
        if self.vis:
            self.vis.destroy_window()
        print("【Viewer】 thread has finished.")

    def _set_cloud(self, cloud, positions, color):
        if not positions:
            cloud.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            cloud.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
        else:
            points = np.asarray(list(positions.values()), dtype=float).reshape(-1, 3)
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
        self.vis.update_geometry(cloud)

    def _replace_cameras(self, stored, poses, size, color):
        for geom in stored.values():
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        stored.clear()
        for frame_id, pose_matrix in poses.items():
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            if color is not None:
                coord_frame.paint_uniform_color(color)
            coord_frame.transform(np.asarray(pose_matrix, dtype=float).reshape(4, 4))
            self.vis.add_geometry(coord_frame, reset_bounding_box=False)
            stored[frame_id] = coord_frame

    def _render_current_scene(self):
        with self.lock:
            self._set_cloud(self.global_cloud, self.global_landmarks, (0.55, 0.55, 0.55))
            self._set_cloud(self.local_cloud, self.landmarks, (1.0, 0.45, 0.05))
            self._replace_cameras(self.global_cameras, self.global_poses, 0.06, (0.35, 0.35, 0.35))
            self._replace_cameras(self.local_cameras, self.poses, 0.12, None)

        if not self.scene_initialized and (
                self.landmarks or self.poses or self.global_landmarks or self.global_poses):
            self.vis.reset_view_point(True)
            self.scene_initialized = True

        self.vis.update_renderer()

    def shutdown(self):
        if self.is_running:
            self.viewer_queue.put(None)
        self.is_running = False
