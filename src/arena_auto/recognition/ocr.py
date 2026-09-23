"""OCR provider abstraction, preprocessing, and power text parsing."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import numpy as np

from arena_auto.exceptions import OCRProviderError
from arena_auto.models.config import OcrSettings
from arena_auto.models.recognition import OCRBox, Region

logger = logging.getLogger(__name__)

OCR_FAILED = None  # explicit sentinel alias - never coerce failures to 0


class OCRProvider(ABC):
    """Pluggable OCR engine interface."""

    name: str = "base"
    supports_read_kwargs: bool = False

    @abstractmethod
    def recognize(self, image: np.ndarray, **kwargs) -> List[OCRBox]:
        """Return zero or more text boxes found in a BGR image."""

    def close(self) -> None:  # optional resource cleanup
        return None


def preprocess_for_ocr(image: np.ndarray, settings: OcrSettings) -> np.ndarray:
    """ROI -> upscale -> grayscale -> contrast -> optional threshold."""
    if image is None or image.size == 0:
        raise ValueError("OCR 输入图像为空")

    processed = image
    scale = max(1, int(settings.scale))
    if scale > 1:
        processed = cv2.resize(
            processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    if processed.ndim == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed

    # mild contrast boost via CLAHE-like stretch
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

    if settings.threshold:
        # Prefer Otsu; fall back to adaptive when result is nearly all black/white.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = float(np.count_nonzero(binary)) / binary.size
        if white_ratio < 0.02 or white_ratio > 0.98:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )
        # keep text dark on light for engines that expect that
        if white_ratio > 0.5:
            binary = cv2.bitwise_not(binary)
        return binary

    return gray


# ---------------------------------------------------------------------------
# Power parsing
# ---------------------------------------------------------------------------

_THOUSAND_SEP_RE = re.compile(r"[,\s.]+")
_ALLOWED_CLEAN_RE = re.compile(r"[^0-9]")
# Screen text is often labeled "战斗力：1,234,567" / "战力 4575674" / "Power: 123"
_POWER_LABEL_RE = re.compile(
    r"^\s*(?:战斗力|战力|Power)\s*[：:．.、]?\s*",
    re.IGNORECASE,
)
# power label prefix, e.g. "战斗力：1,234,567" / "战力 4575674" / "Power: 1234567"
_POWER_LABEL_RE = re.compile(
    r"^(?:战斗力|战力|power)\s*[：:．.、,]?\s*",
    re.IGNORECASE,
)
_OCR_DIGIT_MAP = {
    "I": "1",
    "l": "1",
    "|": "1",
    "i": "1",
    "O": "0",
    "o": "0",
    "D": "0",
    "S": "5",
    "s": "5",
    "B": "8",
    "g": "9",
    "q": "9",
    "Z": "2",
    "z": "2",
}


def _normalize_ocr_digits(text: str) -> str:
    """Replace common OCR letter/digit confusions only in digit context."""
    chars = []
    for ch in text:
        if ch in _OCR_DIGIT_MAP:
            # only substitute when surrounded by digits/separators potential
            chars.append(_OCR_DIGIT_MAP[ch])
        else:
            chars.append(ch)
    return "".join(chars)


def parse_power(raw: Optional[str]) -> Optional[int]:
    """Parse OCR text into an integer power value.

    Accepts bare numbers ("1,234,567") and labeled text
    ("战斗力：1,234,567", "战力 4575674", "Power: 1200000").
    Returns None (OCR_FAILED) when the text cannot be trusted.
    Never returns 0 as a failure placeholder.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # Strip power label prefix so labeled OCR boxes parse the same as bare digits.
    text = _POWER_LABEL_RE.sub("", text, count=1).strip()
    if not text:
        return None

    # Keep digits and thousand separators only.
    # Reject strings that contain unexpected letters after digit mapping.
    normalized = _normalize_ocr_digits(text)
    # Allow separators: comma, space, dot, apostrophe
    if not re.fullmatch(r"[0-9,\s.']+", normalized):
        # maybe pure junk
        digits_only_candidate = _ALLOWED_CLEAN_RE.sub("", normalized)
        if len(digits_only_candidate) < 3:
            return None
        # still has non-digit noise -> untrusted
        return None

    cleaned = _THOUSAND_SEP_RE.sub("", normalized)
    if not cleaned.isdigit():
        return None
    if len(cleaned) > 12:
        return None

    try:
        value = int(cleaned)
    except ValueError:
        return None

    # Heuristic guards against OCR garbage producing absurd values.
    if value <= 0:
        return None
    if value >= 10**11:
        return None
    return value


def parse_challenge_count(raw: Optional[str]) -> Optional[tuple[int, int]]:
    """Parse `4 / 5` style text into (current, maximum)."""
    if raw is None:
        return None
    text = str(raw).strip()
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if not match:
        return None
    try:
        current = int(match.group(1))
        maximum = int(match.group(2))
    except ValueError:
        return None
    if maximum <= 0 or current < 0 or current > maximum:
        return None
    return current, maximum


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class PaddleOCRProvider(OCRProvider):
    """PaddleOCR backend (preferred)."""

    name = "paddle"

    def __init__(self, settings: Optional[OcrSettings] = None) -> None:
        self.settings = settings or OcrSettings()
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError(f"PaddleOCR 不可用: {exc}") from exc

        kwargs = {
            "use_angle_cls": True,
            "lang": self.settings.lang,
            "show_log": False,
        }
        try:
            self._engine = PaddleOCR(**kwargs)
        except TypeError:
            # older/newer paddleocr signature differences
            try:
                self._engine = PaddleOCR(use_angle_cls=True, lang=self.settings.lang)
            except Exception as exc:  # noqa: BLE001
                raise OCRProviderError(f"PaddleOCR 初始化失败: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError(f"PaddleOCR 初始化失败: {exc}") from exc

    def recognize(self, image: np.ndarray) -> List[OCRBox]:
        if image is None or image.size == 0:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = None
        for method_name in ("predict", "ocr"):
            method = getattr(self._engine, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "predict":
                    result = method(image)
                else:
                    result = method(rgb, cls=True)
                break
            except TypeError:
                try:
                    result = method(image)
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.debug("PaddleOCR %s failed: %s", method_name, exc)
            except Exception as exc:  # noqa: BLE001
                logger.debug("PaddleOCR %s failed: %s", method_name, exc)
        return _parse_paddle_result(result)


class TesseractOCRProvider(OCRProvider):
    """pytesseract backend."""

    name = "tesseract"

    def __init__(self, settings: Optional[OcrSettings] = None) -> None:
        self.settings = settings or OcrSettings()
        try:
            import pytesseract  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError(f"pytesseract 不可用: {exc}") from exc
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError(f"Tesseract 未安装或不在 PATH: {exc}") from exc
        self._pytesseract = pytesseract

    def recognize(self, image: np.ndarray) -> List[OCRBox]:
        if image is None or image.size == 0:
            return []
        config = f"--oem 3 --psm 7 -c tessedit_char_whitelist={self.settings.whitelist}"
        try:
            data = self._pytesseract.image_to_data(
                image,
                config=config,
                output_type=self._pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError(f"Tesseract 识别失败: {exc}") from exc

        boxes: List[OCRBox] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 0:
                continue
            x, y, w, h = (
                int(data["left"][i]),
                int(data["top"][i]),
                int(data["width"][i]),
                int(data["height"][i]),
            )
            boxes.append(OCRBox(text=text, confidence=conf / 100.0, bbox=(x, y, w, h)))
        return boxes


class EasyOCRProvider(OCRProvider):
    """easyocr backend with process-wide shared Reader."""

    name = "easyocr"
    supports_read_kwargs = True

    _shared_reader = None
    _shared_key = None
    _shared_lock = __import__("threading").Lock()

    def __init__(self, settings: Optional[OcrSettings] = None) -> None:
        self.settings = settings or OcrSettings()
        try:
            import easyocr  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError(f"easyocr 不可用: {exc}") from exc

        device = str(getattr(self.settings, "device", "auto") or "auto").lower()
        if device in ("cpu", "0", "false", "no"):
            gpu = False
        elif device in ("cuda", "gpu", "1", "true", "yes"):
            gpu = True
        else:
            try:
                import torch  # type: ignore

                gpu = bool(torch.cuda.is_available())
            except Exception:  # noqa: BLE001
                gpu = False
        key = (tuple(["ch_sim", "en"]), bool(gpu))
        try:
            with EasyOCRProvider._shared_lock:
                if (
                    EasyOCRProvider._shared_reader is None
                    or EasyOCRProvider._shared_key != key
                ):
                    EasyOCRProvider._shared_reader = easyocr.Reader(
                        ["ch_sim", "en"], gpu=gpu, verbose=False
                    )
                    EasyOCRProvider._shared_key = key
                    logger.info("EasyOCR Reader 已创建 gpu=%s", gpu)
                self._reader = EasyOCRProvider._shared_reader
                self.device_label = "CUDA" if gpu else "CPU"
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError(f"easyocr 初始化失败: {exc}") from exc

    @classmethod
    def shared_reader_ready(cls) -> bool:
        with cls._shared_lock:
            return cls._shared_reader is not None

    @classmethod
    def reset_shared_reader(cls) -> None:
        with cls._shared_lock:
            cls._shared_reader = None
            cls._shared_key = None

    def recognize(self, image: np.ndarray, **kwargs) -> List[OCRBox]:
        if image is None or image.size == 0:
            return []
        allowed = {
            "decoder",
            "beamWidth",
            "batch_size",
            "workers",
            "allowlist",
            "blocklist",
            "detail",
            "paragraph",
            "contrast_ths",
            "adjust_contrast",
            "image_color",
            "mag_ratio",
            "text_threshold",
            "low_text",
            "link_threshold",
            "canvas_size",
            "x_ths",
            "y_ths",
            "add_margin",
            "slope_ths",
            "ycenter_ths",
            "height_ths",
            "width_ths",
            "y_ths",
            "disable_warning",
        }
        call_kwargs = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if "allowlist" in call_kwargs or kwargs.get("decoder"):
            call_kwargs.setdefault("decoder", kwargs.get("decoder") or "greedy")
            call_kwargs.setdefault("paragraph", False)
        try:
            results = self._reader.readtext(image, **call_kwargs)
        except TypeError:
            # older easyocr without some kwargs
            try:
                results = self._reader.readtext(image)
            except Exception as exc:  # noqa: BLE001
                raise OCRProviderError(f"easyocr 识别失败: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError(f"easyocr 识别失败: {exc}") from exc
        boxes: List[OCRBox] = []
        for item in results:
            if not item or len(item) < 3:
                continue
            quad, text, conf = item[0], item[1], item[2]
            xs = [int(p[0]) for p in quad]
            ys = [int(p[1]) for p in quad]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            boxes.append(
                OCRBox(
                    text=str(text),
                    confidence=float(conf),
                    bbox=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
                )
            )
        return boxes


class NullOCRProvider(OCRProvider):
    """Fails safe when no engine is installed - returns empty results."""

    name = "null"

    def __init__(self, settings: Optional[OcrSettings] = None) -> None:
        self.settings = settings or OcrSettings()
        logger.warning("OCR 引擎不可用，识别将始终失败（安全模式）")

    def recognize(self, image: np.ndarray, **kwargs) -> List[OCRBox]:
        return []


def _parse_paddle_result(result: object) -> List[OCRBox]:
    """Normalize various paddleocr result formats into OCRBox list."""
    boxes: List[OCRBox] = []
    if result is None:
        return boxes

    # paddleocr 2.x: [[ [box, (text, conf)], ... ]] or [ [box, (text, conf)], ...]
    # paddleocr 3.x: list of dict-like objects
    items: List[object]
    if isinstance(result, list):
        items = result
        # unwrap single outer list of pages
        if len(items) == 1 and isinstance(items[0], list):
            items = items[0]
    else:
        items = [result]

    for item in items:
        try:
            if isinstance(item, dict):
                text = str(item.get("rec_text") or item.get("text") or "")
                conf = float(item.get("rec_score") or item.get("score") or 0.0)
                dt_box = item.get("dt_polys") or item.get("dt_boxes") or item.get("box")
                if text and dt_box is not None:
                    xs = [int(p[0]) for p in dt_box]
                    ys = [int(p[1]) for p in dt_box]
                    bbox = (min(xs), min(ys), max(1, max(xs) - min(xs)), max(1, max(ys) - min(ys)))
                    boxes.append(OCRBox(text=text, confidence=conf, bbox=bbox))
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                box = item[0]
                tail = item[1]
                if isinstance(tail, (list, tuple)) and len(tail) >= 1:
                    text = str(tail[0])
                    conf = float(tail[1]) if len(tail) > 1 else 0.0
                elif isinstance(tail, dict):
                    text = str(tail.get("text") or "")
                    conf = float(tail.get("confidence") or tail.get("score") or 0.0)
                else:
                    continue
                if not text:
                    continue
                if box is None:
                    continue
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                bbox = (
                    min(xs),
                    min(ys),
                    max(1, max(xs) - min(xs)),
                    max(1, max(ys) - min(ys)),
                )
                boxes.append(OCRBox(text=text, confidence=conf, bbox=bbox))
        except Exception as exc:  # noqa: BLE001
            logger.debug("parse paddle item failed: %s", exc)
    return boxes


_PROVIDER_FACTORIES = {
    "paddle": PaddleOCRProvider,
    "tesseract": TesseractOCRProvider,
    "easyocr": EasyOCRProvider,
}


def create_base_ocr_provider(
    settings: Optional[OcrSettings] = None,
    allow_fallback: bool = True,
    allow_null: bool = True,
) -> OCRProvider:
    """Create a concrete OCR provider (no OcrEngine wrapper)."""
    settings = settings or OcrSettings()
    preferred = (settings.provider or "paddle").strip().lower()
    order = [preferred] + [name for name in _PROVIDER_FACTORIES if name != preferred]

    errors: List[str] = []
    for name in order:
        factory = _PROVIDER_FACTORIES.get(name)
        if factory is None:
            continue
        try:
            provider = factory(settings)
            if name != preferred:
                logger.warning("OCR 回退到引擎: %s", name)
            return provider
        except OCRProviderError as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("OCR 引擎不可用 (%s): %s", name, exc)
            if not allow_fallback:
                break

    if allow_null:
        return NullOCRProvider(settings)
    raise OCRProviderError("没有可用的 OCR 引擎: " + "; ".join(errors))


def create_ocr_provider(
    settings: Optional[OcrSettings] = None,
    allow_fallback: bool = True,
    allow_null: bool = True,
) -> OCRProvider:
    """Create configured OCR facade (OcrEngine wrapping a shared provider)."""
    settings = settings or OcrSettings()
    try:
        from arena_auto.recognition.ocr_engine import OcrEngine

        engine = OcrEngine(
            provider=None, settings=settings, lazy=True, use_shared=True
        )
        engine.base_factory = lambda: create_base_ocr_provider(  # type: ignore[attr-defined]
            settings, allow_fallback=allow_fallback, allow_null=allow_null
        )
        return engine  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.debug("OcrEngine 包装不可用，直接创建 provider: %s", exc)
        return create_base_ocr_provider(
            settings, allow_fallback=allow_fallback, allow_null=allow_null
        )


def find_text_boxes(
    provider: OCRProvider,
    image: np.ndarray,
    keyword: str,
    min_confidence: float = 0.3,
) -> List[OCRBox]:
    """Return OCR boxes whose text contains keyword."""
    if not keyword:
        return []
    try:
        boxes = provider.recognize(image)
    except OCRProviderError as exc:
        logger.warning("OCR find_text failed: %s", exc)
        return []
    matched: List[OCRBox] = []
    for box in boxes:
        if box.confidence < min_confidence:
            continue
        if keyword in box.text or box.text in keyword:
            matched.append(box)
    return matched


def crop_from_region(
    image: np.ndarray,
    region: Region,
) -> np.ndarray:
    from arena_auto.recognition.screen import crop_region

    return crop_region(image, region)
