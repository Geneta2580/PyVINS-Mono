import numpy as np

class Frame:
    def __init__(self, frame_id, timestamp):
        self.id = frame_id
        self.timestamp = timestamp

        self.image = None
        self.local_pose = None 
        self.global_pose = None
        self.point_cloud = None

        self.visual_features = None # 畸变矫正后的虚拟针孔像素点
        self.visual_feature_ids = None
        self.tracked_raw_features = None # 原始像素点
        self.tracked_undistorted_features = None # 畸变矫正后的虚拟针孔像素点
        self.tracked_feature_ids = None

        self.is_stationary = False
        self.is_keyframe = False

    # 写入类信息(write)
    def set_image(self, image):
        self.image = image

    def add_visual_features(self, visual_features, feature_ids):
        self.visual_features = visual_features
        self.visual_feature_ids = feature_ids

    def add_tracked_features(self, raw_features, undistorted_features, feature_ids):
        self.tracked_raw_features = raw_features
        self.tracked_undistorted_features = undistorted_features
        self.tracked_feature_ids = feature_ids
        
    def set_local_pose(self, local_pose):
        self.local_pose = local_pose

    def set_global_pose(self, global_pose):
        self.global_pose = global_pose

    def set_point_cloud(self, point_cloud, color):
        self.point_cloud = point_cloud
        self.color = color

    def set_is_stationary(self, is_stationary):
        self.is_stationary = is_stationary

    def set_is_keyframe(self, is_keyframe):
        self.is_keyframe = bool(is_keyframe)

    # 读取类信息(read)
    def get_id(self):
        return self.id

    def get_timestamp(self):
        return self.timestamp

    def get_image(self):
        return self.image

    def get_local_pose(self):
        return self.local_pose

    def get_global_pose(self):
        return self.global_pose

    def get_point_cloud(self):
        return self.point_cloud
    
    def get_visual_features(self):
        return self.visual_features

    def get_visual_feature_ids(self):
        return self.visual_feature_ids

    def get_tracked_raw_features(self):
        return self.tracked_raw_features

    def get_tracked_undistorted_features(self):
        return self.tracked_undistorted_features

    def get_tracked_feature_ids(self):
        return self.tracked_feature_ids
    
    def get_is_stationary(self):
        return self.is_stationary

    def get_is_keyframe(self):
        return self.is_keyframe
