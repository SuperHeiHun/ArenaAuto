"""Strict result-label OCR matching, including spaced or split art lettering."""

from __future__ import annotations

import re
from typing import List, Sequence

from arena_auto.models.config import split_ocr_text
from arena_auto.models.recognition import OCRBox


def normalize_ui_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip("！!。.·…?？")


def _exact_label(keyword: str, text: str, allowed_labels: set) -> bool:
    expected = normalize_ui_text(keyword)
    if not expected:
        return False
    if normalize_ui_text(text) == expected:
        return True
    # Some OCR engines put multiple labels (e.g. '胜利 退出') in one box.
    tokens = [normalize_ui_text(token) for token in text.split()]
    return expected in tokens and all(token in allowed_labels for token in tokens)


def find_result_text_boxes(boxes: Sequence[OCRBox], keywords: str,
                          min_confidence: float = 0.3, companions: str = "") -> List[OCRBox]:
    """Never accept '胜利者的竞技场' as the short label '胜利'."""
    wanted = split_ocr_text(keywords)
    allowed_labels = {normalize_ui_text(k) for k in wanted + split_ocr_text(companions)}
    valid = [b for b in boxes if b.confidence >= min_confidence and b.text.strip()]
    matched = [b for b in valid if any(_exact_label(k, b.text, allowed_labels) for k in wanted)]
    # Join spatially adjacent OCR boxes on the same baseline, never arbitrary
    # full-frame strings. Include neighboring words to reject split titles.
    groups: List[List[OCRBox]] = []
    for box in sorted(valid, key=lambda b: b.bbox[0]):
        x, y, w, h = box.bbox
        for group in groups:
            last = group[-1]
            lx, ly, lw, lh = last.bbox
            overlap = min(y + h, ly + lh) - max(y, ly)
            gap = x - (lx + lw)
            if overlap >= min(h, lh) * 0.6 and 0 <= gap <= max(h, lh) * 1.5:
                group.append(box)
                break
        else:
            groups.append([box])
    for group in groups:
        if len(group) < 2:
            continue
        joined = "".join(normalize_ui_text(b.text) for b in group)
        if not any(joined == normalize_ui_text(k) for k in wanted):
            continue
        left = min(b.bbox[0] for b in group)
        top = min(b.bbox[1] for b in group)
        right = max(b.bbox[0] + b.bbox[2] for b in group)
        bottom = max(b.bbox[1] + b.bbox[3] for b in group)
        matched.append(OCRBox(joined, min(b.confidence for b in group), (left, top, right - left, bottom - top)))
    return matched
