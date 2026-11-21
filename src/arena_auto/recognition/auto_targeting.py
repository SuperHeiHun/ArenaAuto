"""Coordinate-free targeting of visible, complete arena cards.

Pure screenshot/OCR analysis: never taps, scrolls, saves user configuration, or
falls back to old coordinates. Currently supports the light horizontal cards.
"""

from __future__ import annotations

import logging
import statistics
import time
from arena_auto.logging.categories import FLOW_TIMING
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.opponent_detector import OpponentDetector, POWER_KEYWORDS, NON_OPPONENT_KEYWORDS
from arena_auto.exceptions import OCRFailedError, StoppedError
from arena_auto.models.config import AppConfig, split_ocr_text
from arena_auto.models.recognition import ChallengeCount, OCRBox, Opponent, Region
from arena_auto.recognition.ocr import OCRProvider, parse_challenge_count, parse_power, preprocess_for_ocr
from arena_auto.recognition.ocr_engine import OcrEngine
from arena_auto.recognition.ocr_pipeline import recognize_frame
from arena_auto.recognition.screen import crop_region
from arena_auto.recognition.result_text import normalize_ui_text

logger = logging.getLogger(__name__)


@dataclass
class _PowerHeader:
    box: OCRBox
    value: Optional[int] = None
    number_region: Optional[Region] = None

    @property
    def cy(self):
        return self.box.bbox[1] + self.box.bbox[3] / 2

    @property
    def bbox(self):
        return self.box.bbox


@dataclass
class OpponentCard:
    opponent: Opponent
    region: Region  # Actual screenshot pixels, NOT reference coordinates.
    identity: str
    target_source: str
    power_verified: bool = False


class AutoTargeting:
    """Combine OCR anchors with visual card edges; verify on fresh frames."""

    def __init__(self, ocr: OCRProvider, config: AppConfig) -> None:
        self.ocr = ocr
        self.config = config
        self.last_cards: List[OpponentCard] = []
        self.last_image: Optional[np.ndarray] = None
        self.last_reason = ""
        self.last_input = "原始截图"
        self.frame_cache = None
        self.debug = None

    def _boxes(self, image: np.ndarray) -> List[OCRBox]:
        try:
            result = (self.frame_cache.recognize(image, self.config.ocr) if self.frame_cache is not None
                      else recognize_frame(self.ocr, image, self.config.ocr))
            self.last_input = result.input_mode
            return [b for b in result.boxes
                    if b.confidence >= self.config.calibration.min_ocr_confidence and b.text.strip()]
        except StoppedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OCRFailedError(f"自动选点 OCR 失败: {exc}") from exc

    def read_challenge(self, image: np.ndarray) -> Optional[ChallengeCount]:
        """Only accept counts next to configured challenge-label anchors."""
        boxes = [b for b in self._boxes(image) if b.confidence >= 0.5]
        keywords = [part for text in self.config.profile.challenge_keywords for part in split_ocr_text(text)]
        anchors = [b for b in boxes if any(normalize_ui_text(k) in normalize_ui_text(b.text) for k in keywords)]
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
            if candidates:
                return None  # Never override ambiguous evidence with a fallback.
            return self._read_challenge_separator(image, boxes, anchors)
        current, maximum = candidates.pop()
        return ChallengeCount(current, maximum)

    def _read_challenge_separator(self, image, boxes, anchors):
        """If '/' was read as '1', require a real slanted separator in the image.

        No global substitution of '1' or guessing 415 == 4/5. Only an anchored
        three-glyph number with a geometric slash can be split and OCR'd anew.
        """
        reader = self.ocr.provider if isinstance(self.ocr, OcrEngine) else self.ocr
        if getattr(reader, "prefers_single_line", False) is not True:
            return None
        results = set()
        for box in boxes:
            x, y, w, h = box.bbox
            if not any(abs((y + h / 2) - (a.bbox[1] + a.bbox[3] / 2)) <= max(h, a.bbox[3]) * 1.5
                       and x >= a.bbox[0] + a.bbox[2] * 0.7
                       and x - (a.bbox[0] + a.bbox[2]) < image.shape[1] * 0.2 for a in anchors):
                continue
            crop = crop_region(image, box.bbox)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for mask in (binary, cv2.bitwise_not(binary)):
                count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                glyphs = [(i, stats[i]) for i in range(1, count)
                          if stats[i, 3] >= h * 0.4 and stats[i, 4] >= 4
                          and stats[i, 0] > 0 and stats[i, 0] + stats[i, 2] < w
                          and stats[i, 1] > 0 and stats[i, 1] + stats[i, 3] < h]
                if len(glyphs) != 3:
                    continue
                glyphs.sort(key=lambda item: item[1][0])
                middle_id, middle = glyphs[1]
                mx, my, mw, mh, area = middle
                if not 0.25 <= mw / mh <= 0.65 or area / (mw * mh) > 0.7:
                    continue
                ys, xs = np.where(labels == middle_id)
                third = max(1, int(mh // 3))
                upper, lower = xs[ys < my + third], xs[ys >= my + mh - third]
                if not len(upper) or not len(lower) or np.mean(upper) - np.mean(lower) < mh * 0.16:
                    continue
                values = []
                for _, glyph in (glyphs[0], glyphs[2]):
                    gx, gy, gw, gh, _ = glyph
                    # A small border avoids clipping single digits for the rec model.
                    digit = crop[max(0, gy - 3):min(h, gy + gh + 3), max(0, gx - 3):min(w, gx + gw + 3)]
                    read = reader.recognize_line(digit)
                    text = "".join(b.text.strip() for b in read if b.confidence >= 0.7)
                    if not text.isdecimal():
                        break
                    values.append(text)
                if len(values) == 2:
                    parsed = parse_challenge_count("/".join(values))
                    if parsed is not None:
                        results.add(parsed)
        if len(results) != 1:
            return None
        return ChallengeCount(*results.pop())

    def scan(self, image: np.ndarray) -> List[OpponentCard]:
        self.last_image = image
        self.last_reason = "未识别到可靠的列表布局"
        self.last_cards = self.detect_cards(image, self._boxes(image))
        if self.last_cards:
            self.last_reason = ""
        return self.last_cards

    @property
    def input_mode(self) -> str:
        """Which OCR input produced the current result (原始/预处理截图)."""
        return self.last_input

    def detect_cards(self, image: np.ndarray, boxes: Sequence[OCRBox]) -> List[OpponentCard]:
        """Do not synthesize missing rows or accept cropped bottom cards."""
        height, width = image.shape[:2]
        keywords = [part for text in self.config.profile.arena_select_keywords for part in split_ocr_text(text)]
        anchors = [b for b in boxes if b.confidence >= 0.5
                   and any(normalize_ui_text(k) in normalize_ui_text(b.text) for k in keywords)]
        if len(anchors) != 1:
            self.last_reason = "没有唯一可靠的选择对手标题"
            return []
        anchor = anchors[0]
        panel_top = anchor.bbox[1] + anchor.bbox[3]
        mapper = CoordinateMapper(self.config.screen.reference_width, self.config.screen.reference_height,
                                  width, height)
        detector = OpponentDetector(self.config, mapper)
        candidates = [c for c in detector.extract_candidates(boxes)
                      if c.bbox[1] > panel_top and c.bbox[0] > anchor.bbox[0]
                      and detector._label_relation(c.box, boxes) in ("own", "near")]
        # Locate rows from the reliable label, NOT from whether every number
        # has high OCR confidence. Unknown power remains None until ROI reads.
        header_boxes = [b for b in boxes if b.confidence >= 0.5
                        and b.bbox[1] > panel_top and b.bbox[0] > anchor.bbox[0]
                        and any(k in b.text for k in POWER_KEYWORDS)
                        and not any(k in b.text for k in ("防御", "방어덱"))]
        headers = []
        for box in sorted(header_boxes, key=lambda b: b.bbox[1]):
            cy = box.bbox[1] + box.bbox[3] / 2
            if headers and abs(cy - headers[-1].cy) < box.bbox[3] * 0.6:
                continue
            nearby = [c for c in candidates if abs(c.cy - cy) <= box.bbox[3] * 1.5
                      and c.bbox[0] >= box.bbox[0] - 5]
            number = max(nearby, key=lambda c: c.box.confidence) if nearby else None
            headers.append(_PowerHeader(box, number.value if number else None, number.bbox if number else None))
        # A label can itself be missed (the first row of the original example).
        # Recover only bare, high-confidence numbers aligned with the established
        # power column/height; card boundaries and fresh ROI reads still govern.
        known_regions = [h.number_region for h in headers if h.number_region is not None]
        if len(headers) >= 2 and len(known_regions) >= 2:
            number_x = statistics.median(r[0] for r in known_regions)
            number_height = statistics.median(r[3] for r in known_regions)
            for box in boxes:
                value = parse_power(box.text)
                x, y, w, h = box.bbox
                cy = y + h / 2
                if (value is None or box.confidence < 0.5 or len(str(value)) < self.config.calibration.min_power_digits
                        or y <= panel_top or abs(x - number_x) > width * 0.03
                        or not number_height * 0.7 <= h <= number_height * 1.6
                        or any(k in box.text for k in NON_OPPONENT_KEYWORDS + POWER_KEYWORDS)
                        or any(abs(cy - header.cy) <= max(h, header.bbox[3]) * 1.5 for header in headers)):
                    continue
                headers.append(_PowerHeader(box, value, box.bbox))
            headers.sort(key=lambda header: header.cy)
        if len(headers) < 2:
            self.last_reason = "可靠的战力标签少于两行"
            return []
        # Keep the dominant aligned label column (exclude unrelated panels).
        median_x = statistics.median(h.bbox[0] for h in headers)
        ordered = [h for h in headers if abs(h.bbox[0] - median_x) <= width * 0.08]
        # Two or more real OCR headers establish row spacing. No 5-slot extrapolation.
        if len(ordered) < 2:
            self.last_reason = "可靠的战力标签列少于两行"
            return []
        distances = [b.cy - a.cy for a, b in zip(ordered, ordered[1:])]
        pitch = float(statistics.median(distances))
        if pitch < height * 0.08 or pitch > height * 0.4:
            self.last_reason = "对手行间距不符合可见卡片布局"
            return []
        if any(abs(d - pitch) > pitch * 0.2 for d in distances):
            self.last_reason = "对手标签缺行或行间距不一致"
            return []
        if ordered[0].bbox[1] - panel_top > pitch * 0.35:
            self.last_reason = "前序对手缺少战力标签，禁止跳过选择"
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max(30, int(width * 0.04)),
                                minLineLength=max(80, int(width * 0.4)), maxLineGap=max(5, int(width * 0.02)))
        if lines is None:
            self.last_reason = "未找到完整卡片边界"
            return []
        number_regions = [h.number_region for h in ordered if h.number_region is not None]
        power_right = statistics.median(r[0] + r[2] for r in number_regions) if number_regions else max(h.bbox[0] + h.bbox[2] for h in ordered)
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
        first_y = ordered[0].bbox[1]
        if first_y - min(b[1] for b in borders) > pitch * 0.35:
            self.last_reason = "前序对手卡片缺少可靠战力标签，禁止跳过选择"
            return []

        def matches_top(y, header):
            return header.bbox[1] - pitch * 0.3 <= y <= header.bbox[1] + 2

        tops_by_row = [[b for b in borders if matches_top(b[1], header)] for header in ordered]
        top_borders = [b for tops in tops_by_row for b in tops]
        if not top_borders:
            return []
        left = int(statistics.median(b[0] for b in top_borders))
        right = min(width - 1, int(np.percentile([b[2] for b in top_borders], 75)))
        card_width = right - left
        if card_width < width * 0.3:
            return []

        def observed_bottom(top):
            return min((y for x1, y, x2 in borders
                        if top + pitch * 0.8 <= y <= top + pitch * 0.98
                        and x1 <= left + card_width * 0.04
                        and x2 >= right - card_width * 0.04), default=None)

        # A pale border (e.g. the top card against a bright background) can
        # disappear at normal contrast. Retry only inside the established card
        # column, without letting weak lines change its horizontal bounds.
        missing_tops = [h.bbox[1] for h, tops in zip(ordered, tops_by_row) if not tops]
        weak_tops = []
        if missing_tops or first_y - panel_top > pitch * 0.35:
            stop = min(height, max([first_y] + missing_tops) + 4)
            weak_tops = self._weak_top_edges(gray, left, right, max(0, panel_top), stop)
            if any(first_y - y > pitch * 0.35 for y in weak_tops):
                self.last_reason = "前序对手卡片缺少可靠战力标签，禁止跳过选择"
                return []

        cards = []
        for index, (cand, tops) in enumerate(zip(ordered, tops_by_row)):
            # A visible horizontal boundary must corroborate each header.
            if tops:
                top = max(b[1] for b in tops)
                bottom = observed_bottom(top)
                if bottom is None:
                    bottom = int(round(top + pitch * 0.92))
            else:
                # A weak top alone is not enough: require an observed strong
                # bottom too, rather than estimating a card from OCR spacing.
                for top in sorted(set(weak_tops), reverse=True):
                    if matches_top(top, cand) and (bottom := observed_bottom(top)) is not None:
                        break
                else:
                    # A banner can completely cover the first row's top edge.
                    # Recover that leading row only when its own OCR power
                    # header supplies the row anchor, later visible rows agree
                    # on the header-to-top inset, the next card top corroborates
                    # the pitch, and this card's lower boundary is visible.
                    # This never extrapolates a missing power row or an interior
                    # gap and cannot make us skip an unknown earlier opponent.
                    if (index != 0 or len(ordered) < 2 or not tops_by_row[1]
                            or cand.bbox[1] - panel_top > pitch * 0.35):
                        continue
                    measured_insets = [
                        header.bbox[1] - max(boundary[1] for boundary in row_tops)
                        for header, row_tops in zip(ordered[1:], tops_by_row[1:])
                        if row_tops
                    ]
                    if not measured_insets:
                        continue
                    top = int(round(cand.bbox[1] - statistics.median(measured_insets)))
                    next_top = max(boundary[1] for boundary in tops_by_row[1])
                    if (not matches_top(top, cand)
                            or abs((next_top - top) - pitch) > pitch * 0.15):
                        continue
                    bottom = observed_bottom(top)
                    if bottom is None:
                        continue
            if top < 0 or bottom >= height - 3 or bottom <= top:
                continue
            region = (left, top, right - left, bottom - top)
            crop = image[top:bottom, left:right]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            light = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 135)
            if float(np.mean(light)) < 0.45:
                continue
            # Names identify the row only. Click geometry comes directly from
            # the same power-ROI calculation used by automatic calibration.
            names = [b for b in boxes if b.confidence >= 0.5 and b.text.strip()
                     and left <= b.bbox[0] + b.bbox[2] / 2 < left + (right - left) * 0.3
                     and top <= b.bbox[1] < top + pitch * 0.25
                     and not any(k in b.text for k in NON_OPPONENT_KEYWORDS + POWER_KEYWORDS)]
            if not names:
                continue
            name = max(names, key=lambda b: b.confidence)
            number_region = cand.number_region
            if number_region is None:
                hx, hy, hw, hh = cand.bbox
                number_x = hx + hw
                if number_x >= right - 5:
                    continue
                number_region = (number_x, hy, right - number_x, hh)
            _, click_point = detector.power_geometry_from_bbox(number_region, pitch)
            target = click_point.as_tuple()
            if not (left < target[0] < right and top < target[1] < bottom):
                continue
            source = "半自动配置：战力 ROI 左侧"
            opponent = Opponent(id=index + 1, power=cand.value, power_region=number_region,
                                click_position=target)
            cards.append(OpponentCard(opponent, region, name.text.strip(), source))
        if not cards:
            self.last_reason = "未找到完整卡片内可靠的主体点击区域"
        return cards

    @staticmethod
    def _weak_top_edges(gray: np.ndarray, left: int, right: int, start: int, stop: int) -> List[int]:
        """Find lower-contrast edges spanning nearly the entire card column."""
        if stop - start < 3:
            return []
        height, width = gray.shape
        card_width = right - left
        edges = cv2.Canny(gray[start:stop, left:right], 35, 80)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max(30, int(width * 0.04)),
                                minLineLength=int(card_width * 0.9), maxLineGap=max(5, int(width * 0.02)))
        if lines is None:
            return []
        return [start + int(round((y1 + y2) / 2)) for ((x1, y1, x2, y2),) in lines
                if abs(y1 - y2) <= max(3, height * 0.005)
                and min(x1, x2) <= card_width * 0.04 and max(x1, x2) >= card_width * 0.96]

    def select(self, screenshot_factory: Callable[[], np.ndarray], threshold: int, stop_event=None,
               *, layout_hint: Optional[np.ndarray] = None):
        """One layout discovery, then short-circuit stable ROI reads on new frames."""
        for attempt in range(max(1, self.config.recognition.retry_count)):
            self._check_stop(stop_event)
            confirmation_frame = None
            try:
                if layout_hint is not None:
                    hint, layout_hint = layout_hint, None  # One attempt, never a persistent cross-round cache.
                    cards = self.scan(hint)
                    started = time.perf_counter()
                    confirmation_frame = screenshot_factory()
                    capture_ms = (time.perf_counter() - started) * 1000.0
                    self._check_stop(stop_event)
                    from arena_auto.recognition.frame_guard import compare_regions, save_guard_evidence

                    ordered = [c.opponent.id for c in cards] == list(range(1, len(cards) + 1))
                    # Validate the whole list-side panel, including title and
                    # edges/tail, not only the ultimately selected card.
                    left = max(0, min((c.region[0] for c in cards), default=0) - 32)
                    region = (left, 0, hint.shape[1] - left, hint.shape[0])
                    started = time.perf_counter()
                    report = compare_regions(hint, confirmation_frame, [region]) if cards and ordered else None
                    unchanged = report is not None and report.unchanged
                    logger.info("布局验证耗时：新截图 %.0fms；画面校验 %.0fms；复用=%s",
                                capture_ms, (time.perf_counter() - started) * 1000.0, bool(unchanged),
                                extra={"log_category": FLOW_TIMING})
                    if unchanged:
                        logger.info("布局复用：新截图已验证整个列表区域，跳过重复整图 OCR")
                        self.last_image = confirmation_frame
                    else:
                        logger.info("布局复用未通过：画面/列表变化，回退新截图整图 OCR")
                        if report is not None:
                            logger.info("布局复用拒绝原因：%s", report.describe())
                            save_guard_evidence(self.debug, hint, confirmation_frame, report, "layout_rejected")
                        else:
                            logger.info("布局复用拒绝原因：%s", "未发现完整卡片" if not cards else "前序卡片不连续")
                        cards = self.scan(confirmation_frame)
                else:
                    cards = self.scan(screenshot_factory())
            except OCRFailedError as exc:
                self.last_reason = str(exc)
                logger.warning("自动选点布局尝试 %s: %s", attempt + 1, exc)
                continue
            self._check_stop(stop_event)
            if not cards:
                logger.warning("自动选点布局尝试 %s: %s", attempt + 1, self.last_reason)
                continue
            logger.info("自动选点布局尝试 %s: 找到 %s 张完整卡片，开始按顺序复核数字 ROI", attempt + 1, len(cards))
            confirmed = []
            latest = self.last_image
            try:
                for expected, card in enumerate(cards, 1):
                    if card.opponent.id != expected:
                        raise OCRFailedError("前序对手卡片识别不完整，禁止跳过选择")
                    value, latest = self._confirm_card(card, screenshot_factory, stop_event,
                                                       first_frame=confirmation_frame)
                    confirmation_frame = None
                    card.opponent.power = value
                    card.power_verified = True
                    confirmed.append(card.opponent)
                    if value <= threshold:
                        self.last_image = latest
                        logger.info("自动选点 #%s 战力 %s 已通过多帧 ROI 校验，跳过后续卡片", card.opponent.id, value)
                        return confirmed, card.opponent
                self.last_image = latest
                return confirmed, None
            except OCRFailedError as exc:
                self.last_reason = str(exc)
                logger.warning("自动选点尝试 %s 未通过: %s；重新定位列表", attempt + 1, exc)
        raise OCRFailedError(f"自动选点重试耗尽（{self.last_reason}），取消点击，不使用旧坐标")

    @staticmethod
    def _check_stop(stop_event):
        if stop_event is not None and stop_event.is_set():
            raise StoppedError("用户停止")

    def _confirm_card(self, card, screenshot_factory, stop_event, *, first_frame=None):
        needed = max(2, self.config.recognition.stable_reads)
        previous, streak = None, 0
        for read in range(max(needed, self.config.recognition.retry_count)):
            self._check_stop(stop_event)
            prefetched = first_frame is not None
            started = time.perf_counter()
            if first_frame is not None:
                image, first_frame = first_frame, None
            else:
                image = screenshot_factory()
            capture_ms = (time.perf_counter() - started) * 1000.0
            if not self.verify_target(image, card.opponent):
                raise OCRFailedError(f"#{card.opponent.id} 卡片画面已变化，可能刷新或滚动")
            started = time.perf_counter()
            value = self._read_power(image, card.opponent.power_region)
            read_ms = (time.perf_counter() - started) * 1000.0
            self._check_stop(stop_event)
            logger.info("自动选点 #%s 数字 ROI 第 %s 次: %s", card.opponent.id, read + 1, value if value is not None else "OCR_FAILED")
            logger.info("战力复核耗时 #%s 第%s次：截图 %.0fms；ROI读取 %.0fms%s",
                        card.opponent.id, read + 1, capture_ms, read_ms,
                        "（首帧已在布局验证时获取）" if prefetched else "",
                        extra={"log_category": FLOW_TIMING})
            streak = streak + 1 if value is not None and value == previous else 1 if value is not None else 0
            previous = value
            if streak >= needed:
                return value, image
        raise OCRFailedError(f"#{card.opponent.id} 数字 ROI 识别失败或战力不稳定")

    def _read_power(self, image, region):
        """Fresh numeric-only ROI recognition; no cache can fake stable reads."""
        try:
            crop = crop_region(image, region)
            if isinstance(self.ocr, OcrEngine):
                boxes, _ = self.ocr.read_roi(image, region, allowlist=self.config.ocr.power_allowlist,
                                            preprocess=True, cache_key=None, force=True)
                if not any(b.confidence >= 0.5 and parse_power(b.text) is not None for b in boxes):
                    logger.info("战力 ROI 预处理未得到可靠数字，重试原图")
                    # Retry the original crop, rather than repeating the same
                    # thresholded input; keep RapidOCR's single-line fast path.
                    boxes2, _ = self.ocr.read_roi(image, region, allowlist=self.config.ocr.power_allowlist,
                                                  preprocess=False, cache_key=None, force=True)
                    if any(b.confidence >= 0.5 and parse_power(b.text) is not None for b in boxes2):
                        boxes = boxes2
            else:
                boxes = self.ocr.recognize(preprocess_for_ocr(crop, self.config.ocr))
                if not any(b.confidence >= 0.5 and parse_power(b.text) is not None for b in boxes):
                    boxes = self.ocr.recognize(crop)
            numeric = [b for b in boxes if b.confidence >= 0.5 and parse_power(b.text) is not None]
            if not numeric:
                return None
            # OCR sometimes splits comma-separated digits into adjacent pieces.
            text = " ".join(b.text for b in sorted(numeric, key=lambda b: b.bbox[0]))
            return parse_power(text)
        except StoppedError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("自动战力 ROI 识别异常: %s", exc)
            return None

    def draw(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        for card in self.last_cards:
            x, y, w, h = card.region
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 220, 0), 2)
            point = card.opponent.click_position
            cv2.drawMarker(result, point, (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.circle(result, point, 12, (255, 0, 255), 2)
            prefix = "ok" if card.power_verified else "raw"
            cv2.putText(result, f"#{card.opponent.id} {prefix}: {card.opponent.power}",
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
