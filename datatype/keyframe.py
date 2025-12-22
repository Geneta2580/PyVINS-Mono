import numpy as np

class KeyFrame:
    def __init__(self, kf_id, timestamp):
        self.id = kf_id
        self.timestamp = timestamp

        self.image = None
        self.local_pose = None 
        self.global_pose = None
        self.point_cloud = None

        self.visual_features = None
        self.visual_feature_ids = None

        self.is_stationary = False

    # 写入类信息(write)
    def set_image(self, image):
        self.image = image

    def add_visual_features(self, visual_features, feature_ids):
        self.visual_features = visual_features
        self.visual_feature_ids = feature_ids
        
    def set_local_pose(self, local_pose):
        self.local_pose = local_pose

    def set_global_pose(self, global_pose):
        self.global_pose = global_pose

    def set_point_cloud(self, point_cloud, color):
        self.point_cloud = point_cloud
        self.color = color

    def set_is_stationary(self, is_stationary):
        self.is_stationary = is_stationary

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
    
    def get_is_stationary(self):
        return self.is_stationary