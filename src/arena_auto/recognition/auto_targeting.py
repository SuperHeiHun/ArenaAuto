"""Experimental coordinate-free targeting of visible, complete arena cards.

Pure screenshot/OCR analysis: never taps, scrolls, saves user configuration, or
falls back to old coordinates. Currently supports the light horizontal cards.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.opponent_detector import OpponentDetector
from arena_auto.exceptions import OCRFailedError, StoppedError
from arena_auto.models.config import AppConfig, split_ocr_text
from arena_auto.models.recognition import ChallengeCount, OCRBox, Opponent, Region
from arena_auto.recognition.ocr import OCRProvider, parse_challenge_count


@dataclass
class OpponentCard:
    opponent: Opponent
    region: Region  # Actual screenshot pixels, NOT reference coordinates.
    identity: str
    target_source: str


class AutoTargeting:
    """Combine OCR anchors with visual card edges; verify on fresh frames."""

    def __init__(self, ocr: OCRProvider, config: AppConfig) -> None:
        self.ocr = ocr
        self.config = config
        self.last_cards: List[OpponentCard] = []
        self.last_image: Optional[np.ndarray] = None

    def _boxes(self, image: np.ndarray) -> List[OCRBox]:
        try:
            return [b for b in self.ocr.recognize(image) if b.confidence >= 0.5 and b.text.strip()]
        except StoppedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OCRFailedError(f"自动选点 OCR 失败: {exc}") from exc

    def read_challenge(self, image: np.ndarray) -> Optional[ChallengeCount]:
        """Only accept counts next to configured challenge-label anchors."""
        boxes = self._boxes(image)
        keywords = [part for text in self.config.profile.challenge_keywords for part in split_ocr_text(text)]
        anchors = [b for b in boxes if any(k in b.text for k in keywords)]
        candidates = set()
        _, width = image.shape[:2]
        for box in boxes:
            parsed = parse_challenge_count(box.text)
            if parsed is None:
                continue
            x, y, w, h = box.bbox
            for anchor in anchors:
                ax, ay, aw, ah = anchor.bbox
                same_row = abs((y + h / 2) - (ay + ah / 2)) <= max(h, ah) * 1.5
                near = abs((x + w / 2) - (ax + aw / 2)) <= width * 0.3
                if same_row and near:
                    candidates.add(parsed)
        if len(candidates) != 1:
            return None
        current, maximum = candidates.pop()
        return ChallengeCount(current, maximum)

    def scan(self, image: np.ndarray) -> List[OpponentCard]:
        self.last_image = image
        self.last_cards = self.detect_cards(image, self._boxes(image))
        return self.last_cards

    def detect_cards(self, image: np.ndarray, boxes: Sequence[OCRBox]) -> List[OpponentCard]:
        """Do not synthesize missing rows or accept cropped bottom cards."""
        height, width = image.shape[:2]
        keywords = [part for text in self.config.profile.arena_select_keywords for part in split_ocr_text(text)]
        anchors = [b for b in boxes if b.confidence >= 0.5 and any(k in b.text for k in keywords)]
        if len(anchors) != 1:
            return []
        anchor = anchors[0]
        panel_top = anchor.bbox[1] + anchor.bbox[3]
        mapper = CoordinateMapper(width, height, width, height)
        detector = OpponentDetector(self.config, mapper)
        candidates = [c for c in detector.extract_candidates(boxes)
                      if c.bbox[1] > panel_top and c.bbox[0] > anchor.bbox[0]
                      and detector._label_relation(c.box, boxes) in ("own", "near")]
        clusters = detector._cluster_by_x(candidates)
        if not clusters:
            return []
        ordered = sorted(max(clusters, key=len), key=lambda c: c.cy)
        # Two or more real OCR headers establish row spacing. No 5-slot extrapolation.
        if len(ordered) < 2:
            return []
        distances = [b.cy - a.cy for a, b in zip(ordered, ordered[1:])]
        pitch = float(statistics.median(distances))
        if pitch < height * 0.08 or pitch > height * 0.4:
            return []
        if any(abs(d - pitch) > pitch * 0.2 for d in distances):
            return []

        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max(30, int(width * 0.04)),
                                minLineLength=max(80, int(width * 0.3)), maxLineGap=max(5, int(width * 0.02)))
        if lines is None:
            return []
        power_right = statistics.median(c.bbox[0] + c.bbox[2] for c in ordered)
        borders = []
        for line in lines:
            x1, y1, x2, y2 = (int(v) for v in line[0])
            left, right = min(x1, x2), max(x1, x2)
            if (abs(y1 - y2) <= max(3, height * 0.005)
                    and left <= anchor.bbox[0] + width * 0.05
                    and right >= power_right - width * 0.03
                    and min(y1, y2) >= panel_top):
                borders.append((left, int(round((y1 + y2) / 2)), right))
        if not borders:
            return []
        if ordered[0].bbox[1] - min(b[1] for b in borders) > pitch * 0.35:
            return []  # An earlier visible card has no recognized power header.
        left = int(statistics.median(b[0] for b in borders))
        right = min(width - 1, int(statistics.median(b[2] for b in borders)))
        if right - left < width * 0.3:
            return []

        cards = []
        for index, cand in enumerate(ordered):
            # A visible horizontal boundary must corroborate each header.
            tops = [b[1] for b in borders if cand.bbox[1] - pitch * 0.3 <= b[1] <= cand.bbox[1] + 2]
            if not tops:
                continue
            top = max(tops)
            bottom = int(round(top + pitch * 0.92))
            # Prefer an observed lower boundary, near the expected card bottom.
            bottoms = [b[1] for b in borders if top + pitch * 0.8 <= b[1] <= top + pitch * 0.98]
            if bottoms:
                bottom = min(bottoms)
            if top < 0 or bottom >= height - 3 or bottom <= top:
                continue
            region = (left, top, right - left, bottom - top)
            crop = image[top:bottom, left:right]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            light = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 135)
            if float(np.mean(light)) < 0.45:
                continue
            # A player-name anchor provides identity and a safe clickable fallback.
            names = [b for b in boxes if b.confidence >= 0.5 and b.text.strip()
                     and left <= b.bbox[0] < left + (right - left) * 0.3
                     and top <= b.bbox[1] < top + pitch * 0.25
                     and not any(k in b.text for k in ("战力", "战斗力", "排名", "等级", "挑战"))]
            if not names:
                continue
            name = max(names, key=lambda b: b.confidence)
            target = (name.bbox[0] + name.bbox[2] // 2, name.bbox[1] + name.bbox[3] // 2)
            source = "卡片名称"
            avatar = self._avatar_center(image, region, target[0])
            if avatar is not None:
                target, source = avatar, "头像"
            if not (left < target[0] < right and top < target[1] < bottom):
                continue
            opponent = Opponent(id=index + 1, power=cand.value, power_region=cand.bbox,
                                click_position=target)
            cards.append(OpponentCard(opponent, region, name.text.strip(), source))
        return cards

    @staticmethod
    def _avatar_center(image: np.ndarray, region: Region, name_x: int):
        left, top, width, height = region
        x0 = max(left, int(name_x - height * 0.6))
        x1 = min(left + int(width * 0.3), int(name_x + height * 0.6))
        y0, y1 = top + int(height * 0.25), top + height
        if x1 <= x0 or y1 <= y0:
            return None
        gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (5, 5), 1), cv2.HOUGH_GRADIENT,
                                  dp=1.2, minDist=max(12, height * 0.5), param1=100, param2=25,
                                  minRadius=max(8, int(height * 0.18)), maxRadius=max(9, int(height * 0.4)))
        if circles is None:
            return None
        valid = [(int(round(x0 + x)), int(round(y0 + y))) for x, y, r in circles[0]
                 if x - r >= 0 and x + r < x1 - x0 and y - r >= 0 and y + r < y1 - y0]
        return valid[0] if len(valid) == 1 else None

    def select(self, screenshot_factory: Callable[[], np.ndarray], threshold: int, stop_event=None):
        """Require stable identity, power and geometry on fresh frames before tap."""
        stable_needed = max(2, self.config.recognition.stable_reads)
        attempts = max(stable_needed, self.config.recognition.retry_count)
        previous = None
        streak = 0
        for _ in range(attempts):
            if stop_event is not None and stop_event.is_set():
                raise StoppedError("用户停止")
            cards = self.scan(screenshot_factory())
            if stop_event is not None and stop_event.is_set():
                raise StoppedError("用户停止")
            if not cards:
                raise OCRFailedError("自动选点未找到可靠的完整对手卡片，禁止使用旧坐标")
            # Do not skip an unrecognized earlier visible row to choose a later one.
            if [c.opponent.id for c in cards] != list(range(1, len(cards) + 1)):
                raise OCRFailedError("前序对手卡片识别不完整，禁止跳过选择")
            selected = next((c for c in cards if c.opponent.power <= threshold), None)
            signature = tuple((c.identity, c.opponent.power) for c in cards)
            geometry = tuple(c.region for c in cards)
            same = previous is not None and previous[0] == signature
            if same:
                tolerance = max(3, int(min(self.last_image.shape[:2]) * 0.015))
                same = all(abs(a - b) <= tolerance for ra, rb in zip(previous[1], geometry) for a, b in zip(ra, rb))
            streak = streak + 1 if same else 1
            previous = (signature, geometry)
            if streak >= stable_needed:
                return [c.opponent for c in cards], selected.opponent if selected else None
        raise OCRFailedError("对手列表正在变化或战力不稳定，取消点击")

    def draw(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        for card in self.last_cards:
            x, y, w, h = card.region
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 220, 0), 2)
            point = card.opponent.click_position
            cv2.drawMarker(result, point, (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.circle(result, point, 12, (255, 0, 255), 2)
            cv2.putText(result, f"#{card.opponent.id}: {card.opponent.power}",
                        (x + 5, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        return result

    def verify_target(self, image: np.ndarray, opponent: Opponent) -> bool:
        """Cheap just-before-tap check: OCR may have taken seconds on CPU."""
        if self.last_image is None or image.shape != self.last_image.shape:
            return False
        card = next((c for c in self.last_cards if c.opponent.id == opponent.id), None)
        if card is None:
            return False
        x, y, w, h = card.region
        px, py = opponent.click_position
        if not (x < px < x + w and y < py < y + h):
            return False
        old = self.last_image[y:y + h, x:x + w]
        new = image[y:y + h, x:x + w]
        # Downsample and smooth to ignore compression / small portrait animations.
        old_small = cv2.GaussianBlur(cv2.resize(old, (128, 32)), (3, 3), 0)
        new_small = cv2.GaussianBlur(cv2.resize(new, (128, 32)), (3, 3), 0)
        return float(np.mean(cv2.absdiff(old_small, new_small))) <= 8.0
