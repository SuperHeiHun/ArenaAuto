"""Discover 1..5 opponent slots from full-frame OCR using scored features."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from arena_auto.calibration.coordinate_mapper import CoordinateMapper
from arena_auto.calibration.models import (
    SOURCE_OCR,
    SOURCE_STRUCTURE,
    CalibrationDetection,
    CalibrationPoint,
    CalibrationRegion,
)
from arena_auto.calibration.roi_detector import RoiDetector
from arena_auto.models.config import AppConfig, CalibrationWeights
from arena_auto.models.recognition import OCRBox
from arena_auto.recognition.ocr import parse_power

logger = logging.getLogger(__name__)

_NEIGHBOR_EXCLUDE_KEYWORDS = (
    "金币",
    "钻石",
    "点券",
    "等级",
    "经验",
    "排行",
    "排名",
    "挑战",
    "战绩",
    "胜场",
    "防御",
)

# Positive labels: prefer OCR boxes like "战斗力：1,234,567" over bare numbers.
_POWER_KEYWORDS = ("战斗力", "战力")


@dataclass
class PowerCandidate:
    box: OCRBox
    value: int
    digits: int
    numeric_score: float
    position_score: float

    @property
    def cx(self) -> float:
        return self.box.bbox[0] + self.box.bbox[2] / 2.0

    @property
    def cy(self) -> float:
        return self.box.bbox[1] + self.box.bbox[3] / 2.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.box.bbox


@dataclass
class SlotHypothesis:
    spacing: float
    anchor_y: float
    matched: List[Tuple[int, PowerCandidate]]
    regularity: float
    score: float
    median_bbox: Tuple[int, int, int, int]
    median_cx: float


class OpponentDetector:
    """Multi-feature opponent structure detection (never `len==5` alone)."""

    def __init__(self, config: AppConfig, mapper: CoordinateMapper) -> None:
        self.config = config
        self.mapper = mapper
        self.roi = RoiDetector(config, mapper)

    # ----------------------------------------------------------- candidates
    def extract_candidates(self, boxes: Sequence[OCRBox]) -> List[PowerCandidate]:
        cal = self.config.calibration
        candidates: List[PowerCandidate] = []
        rejected = 0
        for box in boxes:
            if box.confidence < cal.min_ocr_confidence:
                continue
            value = parse_power(box.text)
            if value is None:
                continue
            digits = len(str(value))
            if digits < cal.min_power_digits or digits > cal.max_power_digits:
                rejected += 1
                continue
            relation = self._label_relation(box, boxes)
            if relation == "exclude":
                rejected += 1
                continue
            candidates.append(
                PowerCandidate(
                    box=box,
                    value=value,
                    digits=digits,
                    numeric_score=self._numeric_score(box.text, digits, relation),
                    position_score=self._position_score(box),
                )
            )
        logger.info(
            "[CALIBRATION] 找到 %s 个数字候选 (排除 %s)",
            len(candidates),
            rejected,
        )
        return candidates

    def _numeric_score(self, raw: str, digits: int, relation: str) -> float:
        base = max(0.4, 1.0 - abs(digits - 7) * 0.1)
        if any(sep in raw for sep in (",", ".", " ")):
            base = min(1.0, base + 0.1)
        # Prefer labeled power text: "战斗力：xxx" in-box, or nearby label box.
        if relation == "own":
            base = min(1.0, base + 0.2)
        elif relation == "near":
            base = min(1.0, base + 0.1)
        return base

    def _position_score(self, box: OCRBox) -> float:
        w = max(1, self.mapper.actual_width)
        h = max(1, self.mapper.actual_height)
        x, y, bw, bh = box.bbox
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        score = 1.0
        if cy < 0.10 or cy > 0.95:
            score -= 0.5
        elif cy < 0.16 or cy > 0.90:
            score -= 0.15
        if cx < 0.08 or cx > 0.92:
            score -= 0.3
        return max(0.2, min(1.0, score))

    def _near_keyword(
        self,
        box: OCRBox,
        boxes: Sequence[OCRBox],
        keywords: Sequence[str],
    ) -> bool:
        bx, by, bw, bh = box.bbox
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0
        reach = max(bw, bh) * 2.5 + 30
        for other in boxes:
            if other is box:
                continue
            ox, oy, ow, oh = other.bbox
            ocx, ocy = ox + ow / 2.0, oy + oh / 2.0
            if abs(ocx - bcx) > reach or abs(ocy - bcy) > reach:
                continue
            if any(kw in other.text for kw in keywords):
                return True
        return False

    def _label_relation(
        self, box: OCRBox, boxes: Sequence[OCRBox]
    ) -> str:
        """Classify keyword context for a power candidate.

        Returns:
            "exclude" — reject (e.g. 金币/排名/防御 label owns this number)
            "own"     — box itself is labeled "战斗力：xxx"
            "near"    — a separate "战斗力" label box is nearest
            ""        — no keyword context
        """
        if any(kw in box.text for kw in _NEIGHBOR_EXCLUDE_KEYWORDS):
            # own text wins even if it also contains 战力
            # (e.g. "防御卡组战斗力" / "挑战战斗力xxx")
            return "exclude"
        if any(kw in box.text for kw in _POWER_KEYWORDS):
            return "own"

        bx, by, bw, bh = box.bbox
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0
        reach = max(bw, bh) * 2.5 + 30
        best_exclude: Optional[float] = None
        best_power: Optional[float] = None
        for other in boxes:
            if other is box:
                continue
            ox, oy, ow, oh = other.bbox
            ocx, ocy = ox + ow / 2.0, oy + oh / 2.0
            if abs(ocx - bcx) > reach or abs(ocy - bcy) > reach:
                continue
            dist = ((ocx - bcx) ** 2 + (ocy - bcy) ** 2) ** 0.5
            # exclude beats power inside the same neighbor box
            # (e.g. "防御卡组战斗力" is both)
            if any(kw in other.text for kw in _NEIGHBOR_EXCLUDE_KEYWORDS):
                if best_exclude is None or dist < best_exclude:
                    best_exclude = dist
            elif any(kw in other.text for kw in _POWER_KEYWORDS):
                if best_power is None or dist < best_power:
                    best_power = dist
        if best_power is not None and (
            best_exclude is None or best_power <= best_exclude
        ):
            return "near"
        if best_exclude is not None:
            return "exclude"
        return ""

    def _near_excluded_keyword(
        self, box: OCRBox, boxes: Sequence[OCRBox]
    ) -> bool:
        return self._label_relation(box, boxes) == "exclude"

    # ------------------------------------------------------------- clusters
    def _cluster_by_x(
        self, candidates: Sequence[PowerCandidate]
    ) -> List[List[PowerCandidate]]:
        if not candidates:
            return []
        tol = self.mapper.pad_to_actual(
            self.config.calibration.x_tolerance, 0
        )[0]
        tol = max(20, tol)
        ordered = sorted(candidates, key=lambda c: c.cx)
        clusters: List[List[PowerCandidate]] = [[ordered[0]]]
        for cand in ordered[1:]:
            current = clusters[-1]
            med_cx = statistics.median(c.cx for c in current)
            if abs(cand.cx - med_cx) <= tol:
                current.append(cand)
            else:
                clusters.append([cand])
        return clusters

    def _spacing_limits(self) -> Tuple[float, float]:
        cal = self.config.calibration
        lo = self.mapper.pad_to_actual(cal.spacing_min, 0)[0]
        hi = self.mapper.pad_to_actual(cal.spacing_max, 0)[0]
        return (max(10.0, float(lo)), max(float(lo) + 10.0, float(hi)))

    def _slot_tolerance(self) -> float:
        return float(
            max(10, self.mapper.pad_to_actual(self.config.calibration.slot_match_tolerance, 0)[0])
        )

    def _hypothesis_for_cluster(
        self, cluster: Sequence[PowerCandidate]
    ) -> Optional[SlotHypothesis]:
        if len(cluster) < 2:
            return None
        lo, hi = self._spacing_limits()
        tol = self._slot_tolerance()
        ys = sorted(c.cy for c in cluster)
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        valid_diffs = [d for d in diffs if lo <= d <= hi]
        spacing_pool = valid_diffs if valid_diffs else [d for d in diffs if d > 1]
        if not spacing_pool:
            return None
        spacing_candidates = [statistics.median(spacing_pool)]
        spacing_candidates.extend(sorted(set(valid_diffs)))
        if len(valid_diffs) >= 2:
            spacing_candidates.append(statistics.mean(valid_diffs))

        best: Optional[SlotHypothesis] = None
        for spacing in spacing_candidates:
            if spacing <= 0:
                continue
            for anchor in ys:
                matched: List[Tuple[int, PowerCandidate]] = []
                used_cands: set = set()
                for slot_index in range(5):
                    expected = anchor + slot_index * spacing
                    if expected < -tol or expected > self.mapper.actual_height + tol:
                        continue
                    best_cand: Optional[PowerCandidate] = None
                    best_dist = tol
                    for cand in cluster:
                        if id(cand) in used_cands:
                            continue
                        dist = abs(cand.cy - expected)
                        if dist <= best_dist:
                            best_dist = dist
                            best_cand = cand
                    if best_cand is not None:
                        matched.append((slot_index, best_cand))
                        used_cands.add(id(best_cand))
                if len(matched) < 2:
                    continue
                hyp = self._score_hypothesis(spacing, anchor, matched, cluster)
                if best is None or hyp.score > best.score or (
                    abs(hyp.score - best.score) < 1e-6
                    and len(hyp.matched) > len(best.matched)
                ):
                    best = hyp
        return best

    def _score_hypothesis(
        self,
        spacing: float,
        anchor: float,
        matched: List[Tuple[int, PowerCandidate]],
        cluster: Sequence[PowerCandidate],
    ) -> SlotHypothesis:
        cands = [c for _, c in matched]
        ys = sorted(c.cy for c in cands)
        if len(ys) >= 2:
            diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            mean = statistics.mean(diffs) if diffs else 1.0
            if mean > 0:
                cv = (statistics.pstdev(diffs) / mean) if len(diffs) > 1 else 0.0
                regularity = max(0.0, min(1.0, 1.0 - cv))
            else:
                regularity = 0.0
        else:
            regularity = 0.0

        structure_score = len(matched) / 5.0
        numeric = statistics.mean(c.numeric_score for c in cands)
        ocr_conf = statistics.mean(c.box.confidence for c in cands)
        med_cx = statistics.median(c.cx for c in cands)
        tol_x = max(20.0, self.mapper.pad_to_actual(self.config.calibration.x_tolerance, 0)[0])
        position_pool = [
            max(0.2, min(1.0, c.position_score - abs(c.cx - med_cx) / (4.0 * tol_x)))
            for c in cands
        ]
        position = statistics.mean(position_pool)

        score = self._weighted_score(
            numeric, position, regularity, structure_score, ocr_conf
        )
        widths = [c.bbox[2] for c in cands]
        heights = [c.bbox[3] for c in cands]
        median_bbox = (
            int(round(med_cx - statistics.median(widths) / 2.0)),
            int(round(statistics.median(c.box.bbox[1] for c in cands))),
            int(statistics.median(widths)),
            int(statistics.median(heights)),
        )
        return SlotHypothesis(
            spacing=spacing,
            anchor_y=anchor,
            matched=matched,
            regularity=regularity,
            score=score,
            median_bbox=median_bbox,
            median_cx=med_cx,
        )

    def _weighted_score(
        self,
        numeric: float,
        position: float,
        spacing_score: float,
        structure_score: float,
        ocr_conf: float,
    ) -> float:
        w: CalibrationWeights = self.config.calibration.weights
        total = (
            w.numeric + w.position + w.spacing + w.structure + w.ocr
        )
        if total <= 0:
            return 0.0
        value = (
            numeric * w.numeric
            + position * w.position
            + spacing_score * w.spacing
            + structure_score * w.structure
            + ocr_conf * w.ocr
        ) / total
        return max(0.0, min(1.0, value))

    # ---------------------------------------------------------------- detect
    def detect(
        self, boxes: Sequence[OCRBox]
    ) -> Tuple[List[CalibrationDetection], List[str]]:
        notes: List[str] = []
        screen_w = max(1, self.mapper.actual_width)
        screen_h = max(1, self.mapper.actual_height)
        candidates = self.extract_candidates(boxes)
        if not candidates:
            notes.append("未发现战力数字候选")
            logger.warning("[CALIBRATION] 找到 0 个对手战力候选")
            return [], notes

        clusters = self._cluster_by_x(candidates)
        hypotheses = []
        for cluster in clusters:
            hyp = self._hypothesis_for_cluster(cluster)
            if hyp is not None:
                hypotheses.append((hyp, cluster))
        logger.info(
            "[CALIBRATION] 对手结构假设数: %s (候选列簇 %s)",
            len(hypotheses),
            len(clusters),
        )

        if hypotheses:
            hyp, cluster = max(
                hypotheses, key=lambda item: (len(item[0].matched), item[0].score)
            )
            notes.append(
                f"重复结构: 间距≈{hyp.spacing:.0f}px, 命中 {len(hyp.matched)}/5, "
                f"score={hyp.score:.2f}"
            )
            logger.info(
                "[CALIBRATION] 对手结构分析成功 spacing=%.1f matched=%s score=%.3f",
                hyp.spacing,
                len(hyp.matched),
                hyp.score,
            )
            detections = self._build_slots(hyp, screen_w, screen_h, notes)
            logger.info(
                "[CALIBRATION] 找到 %s 个对手战力候选", len(detections)
            )
            return detections, notes

        notes.append("未发现稳定重复结构，使用候选回退（请人工确认）")
        logger.warning("[CALIBRATION] 对手结构分析失败，回退到候选排序")
        ordered = sorted(candidates, key=lambda c: c.cy)[:5]
        detections = []
        for idx, cand in enumerate(ordered, start=1):
            detections.append(
                self._detection_from_candidate(
                    idx,
                    cand,
                    confidence=self._weighted_score(
                        cand.numeric_score,
                        cand.position_score,
                        0.0,
                        0.0,
                        cand.box.confidence,
                    )
                    * 0.8,
                    source=SOURCE_OCR,
                    spacing=None,
                    screen_w=screen_w,
                    screen_h=screen_h,
                )
            )
        return detections, notes

    def _build_slots(
        self,
        hyp: SlotHypothesis,
        screen_w: int,
        screen_h: int,
        notes: List[str],
    ) -> List[CalibrationDetection]:
        by_slot = {slot_index: cand for slot_index, cand in hyp.matched}
        detections: List[CalibrationDetection] = []
        tol = self._slot_tolerance()
        for slot_index in range(1, 6):
            expected_y = hyp.anchor_y + (slot_index - 1) * hyp.spacing
            cand = by_slot.get(slot_index - 1)
            if cand is not None:
                conf = self._weighted_score(
                    cand.numeric_score,
                    cand.position_score,
                    hyp.regularity,
                    len(hyp.matched) / 5.0,
                    cand.box.confidence,
                )
                detections.append(
                    self._detection_from_candidate(
                        slot_index,
                        cand,
                        confidence=conf,
                        source=SOURCE_OCR,
                        spacing=hyp.spacing,
                        screen_w=screen_w,
                        screen_h=screen_h,
                    )
                )
            else:
                if expected_y < -tol or expected_y > screen_h + tol:
                    continue
                x, y, w, h = hyp.median_bbox
                synth_bbox = (
                    int(round(hyp.median_cx - w / 2.0)),
                    int(round(expected_y - h / 2.0)),
                    max(1, w),
                    max(1, h),
                )
                conf = min(0.7, hyp.score * 0.7)
                synth = OCRBox(
                    text="",
                    confidence=0.0,
                    bbox=synth_bbox,
                )
                fake = PowerCandidate(
                    box=synth,
                    value=0,
                    digits=0,
                    numeric_score=0.6,
                    position_score=0.6,
                )
                det = self._detection_from_candidate(
                    slot_index,
                    fake,
                    confidence=conf,
                    source=SOURCE_STRUCTURE,
                    spacing=hyp.spacing,
                    screen_w=screen_w,
                    screen_h=screen_h,
                )
                det.value = None
                det.detail = "结构推导（OCR 缺失）"
                notes.append(
                    f"#{slot_index} OCR 缺失，按间距 {hyp.spacing:.0f}px 推导位置"
                )
                logger.info(
                    "[CALIBRATION] #%s power ROI detected (structure)", slot_index
                )
                detections.append(det)
        return detections

    def _detection_from_candidate(
        self,
        slot_id: int,
        cand: PowerCandidate,
        confidence: float,
        source: str,
        spacing: Optional[float],
        screen_w: int,
        screen_h: int,
    ) -> CalibrationDetection:
        bbox = cand.bbox
        power_region = self.roi.power_region_from_bbox(bbox)
        card = self._card_region(power_region, spacing, screen_w, screen_h)
        if source == SOURCE_OCR:
            detail = f"战力：{cand.value:,}" if cand.value else ""
            logger.info(
                "[CALIBRATION] #%s power ROI detected conf=%.2f value=%s",
                slot_id,
                confidence,
                cand.value or "-",
            )
        else:
            detail = "structure"
        return CalibrationDetection(
            name=f"opponent_{slot_id}",
            region=power_region,
            click_point=card.center,
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            value=cand.value if cand.value else None,
            detail=detail,
        )

    def _card_region(
        self,
        power_region: CalibrationRegion,
        spacing: Optional[float],
        screen_w: int,
        screen_h: int,
    ) -> CalibrationRegion:
        pad = self.config.calibration.power_region_padding
        pad_x, pad_y = self.mapper.pad_to_actual(pad.x, pad.y)
        extend = self.mapper.pad_to_actual(
            self.config.calibration.card_extend_left, 0
        )[0]
        extend = max(power_region.width, extend)

        height = power_region.height + 2 * pad_y
        if spacing:
            height = max(height, int(round(spacing * 0.8)))
        height = min(height, screen_h)

        cy = power_region.y + power_region.height // 2.0
        y = int(round(cy - height / 2.0))
        y = max(0, min(y, screen_h - 1))
        height = max(1, min(height, screen_h - y))

        x = max(0, power_region.x - extend)
        width = min(screen_w - x, extend + power_region.width + pad_x)
        width = max(1, width)
        return CalibrationRegion(x=x, y=y, width=width, height=height)
