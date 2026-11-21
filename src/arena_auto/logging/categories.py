"""Log type is independent of severity; never infer it from message text."""

import logging

GENERAL = "general"
ADB_TIMING = "adb_timing"
OCR_TIMING = "ocr_timing"
FLOW_TIMING = "flow_timing"

CATEGORY_LABELS = {
    GENERAL: "常规",
    ADB_TIMING: "ADB耗时",
    OCR_TIMING: "OCR耗时",
    FLOW_TIMING: "流程/识别耗时",
}


def normalize_category(category):
    return category if isinstance(category, str) and category in CATEGORY_LABELS else GENERAL


class CategoryFormatter(logging.Formatter):
    def format(self, record):
        category = normalize_category(getattr(record, "log_category", GENERAL))
        record.category_prefix = f"[{CATEGORY_LABELS[category]}] " if category != GENERAL else ""
        return super().format(record)
