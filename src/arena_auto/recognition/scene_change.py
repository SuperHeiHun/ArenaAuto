"""Conservative signal for one early result OCR probe, never page evidence."""

import cv2
import numpy as np


class SettledSceneChange:
    """Signal a large distributed change followed by a stable next frame.

    At most one signal per battle. Animation cannot bypass OCR cooldowns
    repeatedly, and the signal itself never permits a click or outcome.
    """

    def __init__(self):
        self.previous = None
        self.candidate = False
        self.used = False

    def update(self, image: np.ndarray) -> bool:
        if self.used:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        sample = cv2.GaussianBlur(cv2.resize(gray, (64, 36)), (3, 3), 0)
        if self.previous is None:
            self.previous = sample
            return False
        difference = cv2.absdiff(self.previous, sample)
        self.previous = sample
        if self.candidate and float(difference.mean()) <= 2.0:
            self.used = True
            return True
        self.candidate = float(difference.mean()) >= 12.0 and float(np.mean(difference >= 12)) >= 0.35
        return False
