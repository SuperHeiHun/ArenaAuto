"""Fresh-frame visual validation for previously observed geometry only."""

import cv2
import numpy as np


def regions_unchanged(previous: np.ndarray, current: np.ndarray, regions) -> bool:
    """Require near-identical patches, including local changes within a patch.

    No OCR evidence or power value is produced here. Invalid geometry, changed
    resolution, a moved label/card or a changed local block rejects reuse.
    """
    if previous is None or current is None or previous.shape != current.shape:
        return False
    height, width = current.shape[:2]
    if not regions:
        return False
    for x, y, w, h in regions:
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            return False
        old = previous[y:y + h, x:x + w]
        new = current[y:y + h, x:x + w]
        # Keep colour: similar-luminance replacement portraits must also reject
        # the hint. A small blur tolerates isolated compression noise only.
        diff = cv2.absdiff(cv2.GaussianBlur(old, (3, 3), 0), cv2.GaussianBlur(new, (3, 3), 0))
        if float(diff.mean()) > 1.5:
            return False
        for top in range(0, h, 16):
            for left in range(0, w, 16):
                if float(diff[top:top + 16, left:left + 16].mean()) > 3.0:
                    return False
    return True
