"""Unified OCR engine: shared reader, ROI/number OCR, cache, stats, perf timing.

Design goals (behavior-compatible performance layer):
- EasyOCR Reader is created at most once per process (shared / lazy).
- Runtime hot paths use calibrated ROI + digit allowlist, not full-screen OCR.
- Optional ROI cache skips OCR when the crop is nearly unchanged (absdiff).
- Soft timeout: measures and logs over-budget calls without abandoning threads.
- Never coerces OCR failure to 0 (returns None / empty lists).
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from arena_auto.exceptions import OCRProviderError
from arena_auto.models.config import OcrSettings
from arena_auto.models.recognition import OCRBox, Region
from arena_auto.recognition.ocr import (
    OCRProvider,
    parse_power,
    preprocess_for_ocr,
)
from arena_auto.recognition.screen import crop_region

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("arena_auto.performance")

POWER_ALLOWLIST = "0123456789,."
CHALLENGE_ALLOWLIST = "0123456789/"

# Global switch for [PERF] lines (config.logging.performance)
_perf_enabled = False


def set_performance_logging(enabled: bool) -> None:
    """Enable/disable [PERF] logging from config."""
    global _perf_enabled
    _perf_enabled = bool(enabled)


def perf_enabled() -> bool:
    return _perf_enabled


def perf_log(message: str, elapsed_ms: float) -> None:
    if not _perf_enabled:
        return
    line = f"[PERF] {message} = {elapsed_ms:.1f}ms"
    logger.debug(line)
    perf_logger.info(line)


def resolve_gpu(device: str) -> bool:
    """Map ocr.device auto|cpu|cuda to easyocr gpu flag. Never requires CUDA."""
    normalized = (device or "auto").strip().lower()
    if normalized in ("cpu", "0", "false", "no"):
        return False
    if normalized in ("cuda", "gpu", "1", "true", "yes"):
        return True
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _roi_mean_diff(prev: np.ndarray, cur: np.ndarray) -> float:
    if prev is None or cur is None:
        return 999.0
    if prev.shape != cur.shape:
        return 999.0
    try:
        if prev.ndim != cur.ndim:
            return 999.0
        diff = cv2.absdiff(prev, cur)
        return float(diff.mean())
    except Exception:  # noqa: BLE001
        return 999.0


@dataclass
class OcrPerformanceMonitor:
    """Latency / reliability statistics for OCR calls."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    cache_hits: int = 0
    timeouts: int = 0
    _latencies: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    max_samples: int = 512

    def record(
        self,
        elapsed_ms: float,
        *,
        ok: bool = True,
        cache_hit: bool = False,
        timeout: bool = False,
    ) -> None:
        with self._lock:
            self.calls += 1
            if cache_hit:
                self.cache_hits += 1
            if timeout:
                self.timeouts += 1
            if ok:
                self.successes += 1
            else:
                self.failures += 1
            self._latencies.append(float(elapsed_ms))
            if len(self._latencies) > self.max_samples:
                self._latencies = self._latencies[-self.max_samples :]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            samples = list(self._latencies)
            calls = self.calls
            hits = self.cache_hits
            return {
                "calls": calls,
                "successes": self.successes,
                "failures": self.failures,
                "timeouts": self.timeouts,
                "cache_hits": hits,
                "cache_hit_rate": (hits / calls) if calls else 0.0,
                "avg_ms": statistics.fmean(samples) if samples else 0.0,
                "p50_ms": statistics.median(samples) if samples else 0.0,
                "p95_ms": self._percentile(samples, 0.95),
                "max_ms": max(samples) if samples else 0.0,
            }

    @staticmethod
    def _percentile(samples: Sequence[float], p: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return float(ordered[idx])

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.successes = 0
            self.failures = 0
            self.cache_hits = 0
            self.timeouts = 0
            self._latencies.clear()


class _CacheEntry:
    __slots__ = ("image", "boxes", "text", "stamp")

    def __init__(
        self,
        image: Optional[np.ndarray],
        boxes: List[OCRBox],
        text: str,
        stamp: float,
    ) -> None:
        self.image = image
        self.boxes = boxes
        self.text = text
        self.stamp = stamp


class OcrEngine(OCRProvider):
    """Shared OCR facade used by runtime recognition (ROI-first, cached, timed).

    Implements OCRProvider so existing constructors keep working.
    """

    name = "ocr-engine"
    supports_read_kwargs = True

    _shared_lock = threading.Lock()
    # Shared low-level provider factory result (EasyOCR Reader is process-wide).
    _shared_provider: Optional[OCRProvider] = None
    _shared_key: Optional[Tuple[str, str, str]] = None

    def __init__(
        self,
        provider: Optional[OCRProvider] = None,
        settings: Optional[OcrSettings] = None,
        *,
        lazy: bool = False,
        use_shared: bool = True,
    ) -> None:
        self.settings = settings or OcrSettings()
        self._provider = provider
        self._use_shared = use_shared and provider is None
        self._init_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self.monitor = OcrPerformanceMonitor()
        self._cache: Dict[str, _CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._timeout_ms = max(0, int(getattr(self.settings, "timeout_ms", 1500) or 0))
        self._cache_enabled = bool(getattr(self.settings, "cache_enabled", True))
        self._diff_threshold = float(
            getattr(self.settings, "cache_difference_threshold", 3.0)
        )
        self._ready = provider is not None
        self._device_label = str(getattr(self.settings, "device", "auto") or "auto")
        self.base_factory: Optional[Callable[[], OCRProvider]] = None
        # Cooperative cancel: return immediately once stop_event is set.
        self._cancel_check: Optional[Callable[[], bool]] = None
        if not lazy and self._provider is None and (not use_shared or not self._shared_available()):
            self.ensure_ready()

    def set_cancel_check(self, fn: Optional[Callable[[], bool]]) -> None:
        """Install a stop predicate; when True, recognize/read_roi abort before OCR."""
        self._cancel_check = fn

    def _cancelled(self) -> bool:
        fn = self._cancel_check
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ provider
    def _shared_available(self) -> bool:
        return OcrEngine._shared_provider is not None

    @classmethod
    def shared_provider_key(cls, settings: OcrSettings) -> Tuple[str, str, str]:
        device = str(getattr(settings, "device", "auto") or "auto").lower()
        return (str(settings.provider), str(settings.lang), device)

    @classmethod
    def create_shared(cls, settings: Optional[OcrSettings] = None) -> "OcrEngine":
        """Return a process-wide OcrEngine (one EasyOCR Reader / provider)."""
        settings = settings or OcrSettings()
        engine = cls(provider=None, settings=settings, lazy=True, use_shared=True)
        return engine

    def _make_base_provider(self) -> OCRProvider:
        if self.base_factory is not None:
            return self.base_factory()
        from arena_auto.recognition.ocr import create_base_ocr_provider

        return create_base_ocr_provider(self.settings)

    def ensure_ready(self) -> OCRProvider:
        """Create/reuse the underlying provider exactly once (thread-safe)."""
        if self._provider is not None:
            return self._provider
        with self._init_lock:
            if self._provider is not None:
                return self._provider
            started = time.perf_counter()
            if self._use_shared:
                key = self.shared_provider_key(self.settings)
                with OcrEngine._shared_lock:
                    if (
                        OcrEngine._shared_provider is not None
                        and OcrEngine._shared_key == key
                    ):
                        self._provider = OcrEngine._shared_provider
                    else:
                        self._provider = self._make_base_provider()
                        OcrEngine._shared_provider = self._provider
                        OcrEngine._shared_key = key
            else:
                self._provider = self._make_base_provider()
            self._ready = True
            elapsed = (time.perf_counter() - started) * 1000.0
            perf_log(f"OCR init ({getattr(self._provider, 'name', '?')})", elapsed)
            logger.info(
                "OCR Ready (%s) device=%s init=%.0fms",
                getattr(self._provider, "name", "?"),
                self._device_label,
                elapsed,
            )
            return self._provider

    @property
    def provider(self) -> OCRProvider:
        return self.ensure_ready()

    @property
    def ready(self) -> bool:
        return self._provider is not None

    def close(self) -> None:
        # Shared providers stay alive for process lifetime.
        return None

    # ----------------------------------------------------------------- recognize
    def recognize(self, image: np.ndarray, **kwargs: Any) -> List[OCRBox]:
        """Full-image OCR with timing + soft timeout (thread-safe)."""
        if image is None or getattr(image, "size", 0) == 0:
            return []
        if self._cancelled():
            logger.debug("OCR 已取消（停止请求），跳过 full recognize")
            return []
        provider = self.ensure_ready()
        started = time.perf_counter()
        timeout = False
        try:
            with self._call_lock:
                boxes = self._call_provider(provider, image, kwargs)
            elapsed = (time.perf_counter() - started) * 1000.0
            if self._timeout_ms and elapsed > self._timeout_ms:
                timeout = True
                logger.warning(
                    "OCR timeout budget exceeded: %.0fms > %sms",
                    elapsed,
                    self._timeout_ms,
                )
            self.monitor.record(elapsed, ok=True, timeout=timeout)
            perf_log("OCR full", elapsed)
            return boxes
        except OCRProviderError as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.monitor.record(elapsed, ok=False)
            perf_log("OCR full (failed)", elapsed)
            logger.warning("OCR 失败: %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - started) * 1000.0
            self.monitor.record(elapsed, ok=False)
            logger.warning("OCR 异常: %s", exc)
            return []

    def _call_provider(
        self,
        provider: OCRProvider,
        image: np.ndarray,
        kwargs: Dict[str, Any],
    ) -> List[OCRBox]:
        if kwargs and getattr(provider, "supports_read_kwargs", False):
            try:
                return provider.recognize(image, **kwargs)
            except TypeError:
                pass
        return provider.recognize(image)

    # ---------------------------------------------------------------- ROI / number
    def read_roi(
        self,
        image: np.ndarray,
        region: Region,
        *,
        allowlist: Optional[str] = None,
        detail: bool = False,
        preprocess: bool = True,
        cache_key: Optional[str] = None,
        force: bool = False,
    ) -> Tuple[List[OCRBox], str]:
        """Crop region (reference-scaled by caller) and OCR.

        Returns (boxes, joined_text). Uses cache when cache_key set and crop
        is nearly unchanged. `force=True` bypasses cache (stability re-read).
        """
        if image is None or image.size == 0:
            return [], ""
        if self._cancelled():
            logger.debug("OCR 已取消（停止请求），跳过 ROI")
            return [], ""
        try:
            roi = crop_region(image, region)
        except ValueError:
            logger.warning("ROI 无效: %s", region)
            return [], ""

        if (
            self._cache_enabled
            and cache_key
            and not force
        ):
            cached = self._cache_get(cache_key, roi)
            if cached is not None:
                self.monitor.record(0.05, ok=True, cache_hit=True)
                perf_log(f"OCR cache hit [{cache_key}]", 0.05)
                if detail:
                    return list(cached.boxes), cached.text
                return list(cached.boxes), cached.text

        work = roi
        if preprocess:
            try:
                work = preprocess_for_ocr(roi, self.settings)
            except Exception:  # noqa: BLE001
                work = roi

        kwargs: Dict[str, Any] = {}
        if allowlist and getattr(self.provider, "supports_read_kwargs", False):
            kwargs["allowlist"] = allowlist
            kwargs["decoder"] = "greedy"
            kwargs["paragraph"] = False

        started = time.perf_counter()
        try:
            with self._call_lock:
                boxes = self._call_provider(self.provider, work, kwargs)
                # Fallback: allowlist may yield nothing on odd fonts — retry raw.
                if not boxes and allowlist:
                    boxes = self._call_provider(self.provider, work, {})
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - started) * 1000.0
            self.monitor.record(elapsed, ok=False)
            logger.warning("ROI OCR 异常: %s", exc)
            return [], ""
        elapsed = (time.perf_counter() - started) * 1000.0
        if self._timeout_ms and elapsed > self._timeout_ms:
            self.monitor.record(elapsed, ok=True, timeout=True)
            logger.warning("ROI OCR 超出预算: %.0fms key=%s", elapsed, cache_key)
        else:
            self.monitor.record(elapsed, ok=True)
        text = " ".join(b.text for b in boxes)
        perf_log(f"OCR roi [{cache_key or 'anon'}]", elapsed)
        if cache_key and self._cache_enabled:
            self._cache_put(cache_key, roi, boxes, text)
        return boxes, text

    def read_number(
        self,
        image: np.ndarray,
        region: Region,
        *,
        allowlist: str = POWER_ALLOWLIST,
        cache_key: Optional[str] = None,
        force: bool = False,
    ) -> Optional[int]:
        """Digit-only OCR for power-like fields. None on failure (never 0)."""
        boxes, text = self.read_roi(
            image,
            region,
            allowlist=allowlist,
            preprocess=True,
            cache_key=cache_key,
            force=force,
        )
        if not text.strip() and not boxes:
            return None
        # Prefer slash / number candidates then joined text
        candidates = [b.text for b in boxes] + [text]
        for raw in candidates:
            value = parse_power(raw)
            if value is not None:
                return value
        return None

    def read_power(
        self,
        image: np.ndarray,
        region: Region,
        *,
        cache_key: Optional[str] = None,
        force: bool = False,
    ) -> Optional[int]:
        """Power ROI with allowlist + raw fallback. None means OCR_FAILED."""
        boxes, text = self.read_roi(
            image,
            region,
            allowlist=POWER_ALLOWLIST,
            preprocess=True,
            cache_key=cache_key,
            force=force,
        )
        for raw in [b.text for b in boxes] + [text]:
            value = parse_power(raw)
            if value is not None:
                return value
        if force or not boxes:
            # second chance without allowlist (labels like 战力：)
            boxes2, text2 = self.read_roi(
                image,
                region,
                allowlist=None,
                preprocess=True,
                cache_key=None if force else f"{cache_key}#raw" if cache_key else None,
                force=True,
            )
            for raw in [b.text for b in boxes2] + [text2]:
                value = parse_power(raw)
                if value is not None:
                    return value
        return None

    def read_text(
        self,
        image: np.ndarray,
        allowlist: Optional[str] = None,
        detail: bool = False,
    ) -> List[OCRBox]:
        kwargs: Dict[str, Any] = {}
        if allowlist and getattr(self.provider, "supports_read_kwargs", False):
            kwargs["allowlist"] = allowlist
            kwargs["decoder"] = "greedy"
        return self.recognize(image, **kwargs)

    def read_many_rois(
        self,
        image: np.ndarray,
        regions: Sequence[Region],
        *,
        allowlist: Optional[str] = None,
        key_prefix: str = "roi",
        force: bool = False,
    ) -> List[Tuple[List[OCRBox], str]]:
        """Strategy A: sequential ROI OCR on one shared frame (no re-capture)."""
        out: List[Tuple[List[OCRBox], str]] = []
        for index, region in enumerate(regions, start=1):
            out.append(
                self.read_roi(
                    image,
                    region,
                    allowlist=allowlist,
                    cache_key=f"{key_prefix}:{index}",
                    force=force,
                )
            )
        return out

    # -------------------------------------------------------------------- cache
    def _cache_get(self, key: str, roi: Optional[np.ndarray]) -> Optional[_CacheEntry]:
        with self._cache_lock:
            entry = self._cache.get(key)
        if entry is None:
            return None
        if roi is None:
            return entry
        if _roi_mean_diff(entry.image, roi) <= self._diff_threshold:
            return entry
        return None

    def _cache_put(
        self, key: str, roi: np.ndarray, boxes: List[OCRBox], text: str
    ) -> None:
        try:
            snapshot = roi.copy()
        except Exception:  # noqa: BLE001
            snapshot = None
        with self._cache_lock:
            self._cache[key] = _CacheEntry(snapshot, list(boxes), text, time.time())
            # bound memory
            if len(self._cache) > 128:
                oldest = sorted(self._cache.items(), key=lambda kv: kv[1].stamp)
                for old_key, _ in oldest[: len(self._cache) - 128]:
                    self._cache.pop(old_key, None)

    def clear_cache(self, prefix: Optional[str] = None) -> None:
        with self._cache_lock:
            if prefix is None:
                self._cache.clear()
            else:
                for key in [k for k in self._cache if k.startswith(prefix)]:
                    self._cache.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
        stats = self.monitor.get_stats()
        stats["cache_size"] = len(self._cache)
        stats["provider"] = getattr(self._provider, "name", "lazy")
        stats["device"] = self._device_label
        stats["timeout_ms"] = self._timeout_ms
        return stats

    def describe(self) -> str:
        s = self.get_stats()
        return (
            f"OCR 平均：{s['avg_ms']:.0f} ms | "
            f"P95：{s['p95_ms']:.0f} ms | "
            f"缓存命中：{s['cache_hit_rate'] * 100:.0f}%"
        )


def get_ocr_engine(
    settings: Optional[OcrSettings] = None,
    provider: Optional[OCRProvider] = None,
    *,
    lazy: bool = False,
) -> OcrEngine:
    """Helper for call sites that need an OcrEngine."""
    return OcrEngine(provider=provider, settings=settings, lazy=lazy)


def timed_call(label: str, fn: Callable[[], Any]) -> Any:
    """Run fn and emit [PERF] timing when performance logging is on."""
    started = time.perf_counter()
    try:
        return fn()
    finally:
        perf_log(label, (time.perf_counter() - started) * 1000.0)
