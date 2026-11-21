"""Narrow, fail-closed PREPARE reuse for the observed two-panel layout.

All geometry is discovered from pixels and already cached OCR. No old frame
OCR, fixed device coordinates, power confirmation or click authorization here.
Unrecognized layouts retain the controller's full-page validation path.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from arena_auto.recognition.frame_guard import FrameGuardReport, compare_regions
from arena_auto.recognition.ocr import parse_power


def _contains(region, box):
    x, y, w, h = region
    bx, by, bw, bh = box
    return x <= bx and y <= by and bx + bw <= x + w and by + bh <= y + h


def _panels(image, button):
    """Bright connected panel backgrounds, with thin animation bridges removed."""
    height, width = image.shape[:2]
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    # Equivalent per-channel threshold in OpenCV avoids NumPy's full-frame
    # channel reduction overhead on every fresh screenshot.
    mask = cv2.inRange(image, (170, 170, 170), (255, 255, 255))
    size = max(3, int(round(min(width, height) * .01)) | 1)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    candidates = []
    for x, y, w, h, area in stats[1:]:
        if (x >= width * .6 and width * .18 <= w <= width * .4
                and h >= height * .65 and area >= w * h * .15):
            # A thin connection to the bright scene above is not panel geometry.
            rows = np.flatnonzero(mask[y:y + h, x:x + w].mean(axis=1) >= .75 * 255)
            if len(rows):
                top = int(rows[0])
                region = (int(x), int(y + top), int(w), int(h - top))
                if (region[1] <= height * .25 and region[3] >= height * .65
                        and _contains(region, button)
                        and button[1] >= region[1] + region[3] * .75):
                    candidates.append(region)
    if len(candidates) != 1:
        return None
    panel = candidates[0]
    summaries = []
    for x, y, w, h, area in stats[1:]:
        if (width * .3 <= x < panel[0] and y >= height * .75
                and width * .18 <= w <= width * .4 and height * .08 <= h <= height * .25
                and x + w <= panel[0] and 0 <= panel[0] - (x + w) <= width * .04
                and area >= w * h * .15):
            summaries.append(tuple(map(int, (x, y, w, h))))
    return (panel, summaries[0]) if len(summaries) == 1 else None


@dataclass(frozen=True)
class PrepareStructure:
    panel: tuple
    summary: tuple
    button: tuple
    anchors: tuple
    headings: tuple

    @classmethod
    def discover(cls, image, boxes, button):
        if image is None or not boxes:
            return None
        panels = _panels(image, button)
        if panels is None:
            return None
        panel, summary = panels
        height, width = image.shape[:2]
        reliable = [b for b in boxes if b.confidence >= .5]
        # Several card numbers, a populated opponent summary and page heading
        # are required. A bright popup with a button is not a PREPARE layout.
        numbers = [b for b in reliable if _contains(panel, b.bbox) and parse_power(b.text) is not None]
        summary_boxes = [b for b in reliable if _contains(summary, b.bbox)]
        headings = [b for b in reliable if b.bbox[1] + b.bbox[3] < height * .12
                    and b.bbox[0] + b.bbox[2] < width * .6
                    and sum(c.isalpha() for c in b.text) >= 3]
        if len(numbers) < 3 or len(summary_boxes) < 2 or not headings:
            return None
        # Retain non-animated cached UI labels, including own formation counters
        # and navigation, without treating moving character-level labels as UI.
        anchors = []
        heading_regions = []
        for box in reliable:
            x, y, w, h = box.bbox
            in_character_field = (width * .12 <= x + w / 2 <= panel[0] * .85
                                  and height * .12 <= y + h / 2 < summary[1])
            if (not _contains(panel, box.bbox) and not _contains(summary, box.bbox)
                    and (box in headings or (not in_character_field and len(box.text.strip()) >= 3))):
                pad = max(1, int(round(height * .003)))
                left, top = max(0, x - pad), max(0, y - pad)
                right, bottom = min(width, x + w + pad), min(height, y + h + pad)
                region = (left, top, right - left, bottom - top)
                if box in headings:
                    heading_regions.append(region)
                else:
                    anchors.append(region)
        if not heading_regions:
            return None
        return cls(panel, summary, button, tuple(anchors), tuple(heading_regions))

    def validate(self, previous, current):
        """Return (visual report, block_click). All anchors keep strict thresholds."""
        if previous.shape != current.shape:
            return FrameGuardReport(False, "截图尺寸变化"), False
        # Opaque/dialog-like boxes in the character field are not animation.
        # Check both frames: a modal already present in cached evidence is unsafe.
        if _modal_rectangle(previous, self.panel, self.summary) or _modal_rectangle(current, self.panel, self.summary):
            return FrameGuardReport(False, "角色区域出现弹窗样矩形"), True
        if _rectangular_change(previous, current, self.panel, self.summary):
            return FrameGuardReport(False, "角色区域出现矩形覆盖变化，疑似弹窗"), True
        if _panel_dimmed(previous, current, self.panel):
            return FrameGuardReport(False, "稳定面板大范围变暗，疑似弹窗遮罩", self.panel), True
        panels = _panels(current, self.button)
        if panels != (self.panel, self.summary):
            return FrameGuardReport(False, "备战面板结构缺失或位置变化"), False
        # Rounded panel corners expose animated scenery. Check all straight
        # edges and the interior, omitting only the four small corner squares.
        x, y, w, h = self.panel
        radius = max(3, int(round(min(w, h) * .04)))
        panel_regions = [(x + radius, y, w - radius * 2, radius),
                         (x, y + radius, w, h - radius * 2),
                         (x + radius, y + h - radius, w - radius * 2, radius)]
        report = compare_regions(previous, current, [*panel_regions, self.summary, self.button, *self.anchors])
        if report.unchanged:
            for region in self.headings:
                report = compare_regions(previous, current, [region])
                if not report.unchanged and not _heading_outline_matches(previous, current, region):
                    return report, False
            report = FrameGuardReport(True, "备战结构和页面标题已验证")
        return report, False


def _panel_dimmed(previous, current, panel):
    x, y, w, h = panel
    old = previous[y:y + h, x:x + w].mean(axis=2)
    new = current[y:y + h, x:x + w].mean(axis=2)
    return float(((old > 150) & (new < old * .8)).mean()) > .4


def _heading_outline_matches(previous, current, region):
    """White heading glyphs may wobble ~one pixel, not change their identity.

    This is limited to cached heading boxes, never numeric or button evidence.
    Both directions require 99.5% outline coverage and near-equal glyph area.
    Non-white/non-text-like headings simply retain the strict pixel fallback.
    """
    x, y, w, h = region
    old = (previous[y:y + h, x:x + w].min(axis=2) > 190).astype(np.uint8)
    new = (current[y:y + h, x:x + w].min(axis=2) > 190).astype(np.uint8)
    old_count, new_count = int(old.sum()), int(new.sum())
    if not (40 <= old_count < w * h * .45 and .95 <= new_count / old_count <= 1.05):
        return False
    if cv2.connectedComponents(old, 8)[0] < 4 or cv2.connectedComponents(new, 8)[0] < 4:
        return False
    tolerance = max(1.0, previous.shape[0] / 720 * 1.5)
    to_old = cv2.distanceTransform(1 - old, cv2.DIST_L2, 5)
    to_new = cv2.distanceTransform(1 - new, cv2.DIST_L2, 5)
    return (float((to_new[old > 0] <= tolerance).mean()) >= .995
            and float((to_old[new > 0] <= tolerance).mean()) >= .995)


def _modal_rectangle(image, panel, summary):
    """Conservative axis-aligned dialog borders in the central character field.

    Uncertain rectangles reject reuse rather than being whitelisted as animation.
    Stable panels outside that field are verified separately, including dimming.
    """
    height, width = image.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 60, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w / 2, y + h / 2
        if not (width * .12 < cx < panel[0] * .85 and height * .15 < cy < summary[1]
                and w * h >= width * height * .004 and w >= width * .06 and h >= height * .04):
            continue
        polygon = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * .02, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        # A formation platform is a trapezoid, not an axis-aligned dialog.
        corners = [(x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1)]
        if all(any(abs(int(px) - qx) <= w * .06 and abs(int(py) - qy) <= h * .06
                   for qx, qy in corners) for ((px, py),) in polygon):
            return True
    return False


def _rectangular_change(previous, current, panel, summary):
    """Catch low-contrast filled overlays that Canny may not outline."""
    height, width = current.shape[:2]
    left, top = int(width * .12), int(height * .15)
    right, bottom = int(panel[0] * .85), summary[1]
    old = previous[top:bottom, left:right]
    new = current[top:bottom, left:right]
    diff = cv2.absdiff(cv2.GaussianBlur(old, (3, 3), 0), cv2.GaussianBlur(new, (3, 3), 0))
    changed = (diff.mean(axis=2) > 3.0).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(changed, 8)
    for _, _, w, h, area in stats[1:]:
        if (w >= width * .06 and h >= height * .04 and w * h >= width * height * .004
                and area / (w * h) >= .85):
            return True
    return False
