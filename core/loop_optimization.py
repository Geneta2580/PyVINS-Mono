"""Loop-closure optimization integration point.

SALAD currently produces place-recognition candidates in ``loop_detector``.
Geometric verification and pose-graph factor insertion intentionally remain
unimplemented until their measurement and covariance contracts are defined.
"""


class LoopOptimization:
    """Placeholder owned by the estimator in a future implementation."""

    def __init__(self, config):
        self.config = config

    def add_detection(self, detection):
        """Return False until geometric verification/optimization is added."""
        return False
