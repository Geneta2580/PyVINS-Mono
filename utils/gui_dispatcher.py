"""Single-owner OpenCV HighGUI dispatcher.

Worker threads only publish their newest frame.  The main thread owns every
imshow/waitKey call, because OpenCV HighGUI event loops are not thread-safe.
"""

import threading

import cv2


class GuiDispatcher:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending = {}
        self._has_windows = False
        self.enabled = True

    def publish(self, window_name, image):
        """Replace the pending image for one window without blocking a worker."""
        if not self.enabled or image is None:
            return False
        with self._lock:
            self._pending[str(window_name)] = image
        return True

    def show_pending(self):
        """Display pending frames.  This method must only run in main thread."""
        if not self.enabled:
            return
        with self._lock:
            pending = self._pending
            self._pending = {}
        try:
            for window_name, image in pending.items():
                cv2.imshow(window_name, image)
                self._has_windows = True
            if self._has_windows:
                cv2.waitKey(1)
        except cv2.error as exc:
            self.enabled = False
            with self._lock:
                self._pending.clear()
            print(f"【GUI】: visualization disabled: {exc}")

    def close(self):
        if self._has_windows:
            cv2.destroyAllWindows()

