"""Loop-closure optimization integration point.

``loop_detector`` currently produces SALAD candidates and deduplicated
KLT-BRIEF/FAST-BRIEF correspondences.  Epipolar geometric verification,
relative-pose estimation, and pose-graph factor insertion intentionally remain
unimplemented until their measurement and covariance contracts are defined.
"""


class LoopOptimization:
    """Placeholder owned by the estimator in a future implementation."""

    def __init__(self, config):
        self.config = config

    def add_detection(self, detection):
        """Return False until geometric verification/optimization is added."""
        return False
