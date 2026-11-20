"""Optional multilingual RapidOCR ONNX backend with verified DirectML sessions."""

from __future__ import annotations

import logging
import math
import platform
import threading
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from arena_auto.exceptions import OCRProviderError
from arena_auto.models.config import OcrSettings, normalize_ocr_language
from arena_auto.models.recognition import OCRBox
from arena_auto.recognition.ocr import OCRProvider
from arena_auto.paths import get_resource_path, get_writable_path
from arena_auto.recognition.model_store import KOREAN_MODELS, valid_model, download_model

logger = logging.getLogger(__name__)


class RapidOCRProvider(OCRProvider):
    """Detection and recognition use DirectML; CPU fallback is explicit in logs.

    RapidOCR does not support an allowlist decoder. Do not strip Chinese text
    into digits: callers retain their strict power parsing and stable ROI reads.
    """

    name = "rapidocr"
    supports_read_kwargs = False
    prefers_single_line = True

    def __init__(self, settings: Optional[OcrSettings] = None) -> None:
        self.settings = settings or OcrSettings()
        self._lock = threading.RLock()  # Shared DirectML sessions cannot run concurrently.
        self.device_label = "CPU"
        self._gpu_active = False
        try:
            import onnxruntime as ort
            import rapidocr
            from rapidocr import RapidOCR
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError("RapidOCR GPU 依赖未安装；请在当前 Python 环境安装 requirements-gpu.txt") from exc
        self._ort = ort
        self.available_providers = list(ort.get_available_providers())
        device = str(self.settings.device or "auto").strip().lower()
        if device not in ("auto", "cpu", "directml", "dml"):
            raise OCRProviderError("RapidOCR 支持 auto/cpu/directml；NVIDIA CUDA 请使用 EasyOCR 后端")
        wants_gpu = device != "cpu"
        available = platform.system() == "Windows" and "DmlExecutionProvider" in self.available_providers
        if wants_gpu and not available:
            message = "当前运行时没有可用 DirectML（需要 Windows 10 1903+ 和 onnxruntime-directml）"
            if not self.settings.gpu_fallback_to_cpu:
                raise OCRProviderError(message)
            logger.warning("%s；RapidOCR 回退 CPU", message)

        language = normalize_ocr_language(self.settings.lang)
        if language not in ("ch", "japan", "korean"):
            raise OCRProviderError(f"RapidOCR 不支持语言配置: {language!r}；请选择 ch/japan/korean")

        # Chinese/Japanese share PP-OCRv6 multi-language models. Korean uses
        # PP-OCRv5's dedicated Hangul recognizer and Chinese text detector.
        model_dir = Path(rapidocr.__file__).resolve().parent / "models"
        self.language = language
        if language == "korean":
            bundled_dir = get_resource_path("resources/models/rapidocr")
            self._config_model_dir = get_writable_path("resources/models/rapidocr")
            self._model_paths = []
            for filename, url, digest in KOREAN_MODELS:
                bundled_path = bundled_dir / filename
                cached_path = self._config_model_dir / filename
                if valid_model(bundled_path, digest):
                    path = bundled_path
                elif valid_model(cached_path, digest):
                    path = cached_path
                else:
                    path = download_model(cached_path, url, digest)
                self._model_paths.append(path)
        else:
            self._model_paths = [model_dir / "PP-OCRv6_det_small.onnx", model_dir / "PP-OCRv6_rec_small.onnx"]
        if any(not path.is_file() for path in self._model_paths):
            raise OCRProviderError("RapidOCR 模型缺失；请使用 requirements-gpu.txt 中固定的 rapidocr 版本")
        try:
            self._engine = RapidOCR(params={
                "Global.use_cls": False,
                "Global.text_score": 0.3,
                "Global.log_level": "warning",
                "Global.model_root_dir": str(self._config_model_dir) if language == "korean" else str(model_dir),
                "Det.model_path": str(self._model_paths[0]),
                "Rec.model_path": str(self._model_paths[1]),
                "EngineConfig.onnxruntime.use_dml": False,
                "EngineConfig.onnxruntime.use_cuda": False,
                "EngineConfig.onnxruntime.intra_op_num_threads": 4,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
            from rapidocr.utils.typings import LangDet, LangRec, ModelType, OCRVersion

            self._engine.cfg.Rec.lang_type = {
                "ch": LangRec.CH, "japan": LangRec.JAPAN, "korean": LangRec.KOREAN,
            }[language]
            if language == "korean":
                self._engine.cfg.Det.ocr_version = OCRVersion.PPOCRV5
                self._engine.cfg.Det.lang_type = LangDet.CH
                self._engine.cfg.Det.model_type = ModelType.MOBILE
                self._engine.cfg.Rec.ocr_version = OCRVersion.PPOCRV5
                self._engine.cfg.Rec.model_type = ModelType.MOBILE
            elif language == "japan":
                # V6 recognition is multilingual, and its detector is routed
                # through the corresponding multi-language detector family.
                self._engine.cfg.Det.lang_type = LangDet.MULTI
            # Load metadata/character lists using CPU first, then replace both
            # inference sessions with correctly configured DirectML sessions.
            models = [self._engine._load_det_model(), self._engine._load_rec_model()]
            self._wrappers = [model.session for model in models]
            self._cpu_sessions = [wrapper.session for wrapper in self._wrappers]
            if wants_gpu and available:
                try:
                    self._enable_directml()
                except Exception as exc:  # noqa: BLE001
                    if not self.settings.gpu_fallback_to_cpu:
                        raise OCRProviderError(f"DirectML GPU 初始化失败: {exc}") from exc
                    logger.warning("DirectML GPU 初始化失败，RapidOCR 回退 CPU: %s", exc)
            logger.info("RapidOCR 初始化完成 lang=%s device=%s sessions=%s", language, self.device_label, self.session_providers)
        except OCRProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OCRProviderError(f"RapidOCR 初始化失败: {exc}") from exc

    def _enable_directml(self) -> None:
        sessions = []
        for path in self._model_paths:
            options = self._ort.SessionOptions()
            options.enable_mem_pattern = False
            options.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_cpu_mem_arena = False
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            session = self._ort.InferenceSession(str(path), sess_options=options, providers=[
                ("DmlExecutionProvider", {"device_id": str(self.settings.gpu_device_id)}),
                "CPUExecutionProvider",
            ])
            session.disable_fallback()
            if not session.get_providers() or session.get_providers()[0] != "DmlExecutionProvider":
                raise OCRProviderError("ONNX 会话没有实际启用 DirectML，不能标记为 GPU")
            sessions.append(session)
        # Publish only after BOTH sessions are ready; partial failure stays CPU.
        for wrapper, session in zip(self._wrappers, sessions):
            wrapper.session = session
        self._gpu_active = True
        self.device_label = f"DirectML GPU {self.settings.gpu_device_id}"

    @property
    def session_providers(self):
        return {name: list(wrapper.session.get_providers())
                for name, wrapper in zip(("detection", "recognition"), self._wrappers)}

    def _fallback_cpu(self) -> None:
        for wrapper, session in zip(self._wrappers, self._cpu_sessions):
            wrapper.session = session
        self._gpu_active = False
        self.device_label = "CPU"
        logger.warning("OCR Ready (rapidocr) device=CPU：DirectML 推理失败，已回退 CPU")

    def recognize(self, image: np.ndarray, **kwargs) -> List[OCRBox]:
        return self._recognize(image, single_line=False)

    def recognize_line(self, image: np.ndarray) -> List[OCRBox]:
        """An already-located number ROI needs recognition, not text detection."""
        return self._recognize(image, single_line=True)

    def _recognize(self, image: np.ndarray, *, single_line: bool) -> List[OCRBox]:
        if image is None or image.size == 0:
            return []
        work = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        if work.ndim != 3 or work.shape[2] not in (3, 4):
            raise OCRProviderError("RapidOCR 输入必须是灰度/BGR/BGRA 图片")
        if work.shape[2] == 4:
            work = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)
        with self._lock:
            try:
                result = self._engine(work, use_det=not single_line, use_cls=False)
            except Exception as exc:  # noqa: BLE001
                if not self._gpu_active or not self.settings.gpu_fallback_to_cpu:
                    raise OCRProviderError(f"RapidOCR 识别失败: {exc}") from exc
                logger.warning("DirectML 推理错误: %s", exc)
                self._fallback_cpu()
                try:
                    result = self._engine(work, use_det=not single_line, use_cls=False)
                except Exception as retry_exc:  # noqa: BLE001
                    raise OCRProviderError(f"RapidOCR CPU 回退识别失败: {retry_exc}") from retry_exc
            if any(getattr(result, name, None) is None for name in ("txts", "scores")):
                return []
            height, width = work.shape[:2]
            quads = getattr(result, "boxes", None)
            if single_line:
                quads = [np.array([[0, 0], [width, 0], [width, height], [0, height]]) for _ in result.txts]
            elif quads is None:
                return []
            boxes = []
            for quad, text, score in zip(quads, result.txts, result.scores):
                points = np.asarray(quad, dtype=float)
                confidence = float(score)
                if points.shape != (4, 2) or not np.isfinite(points).all() or not math.isfinite(confidence) or not str(text).strip():
                    continue
                x0 = max(0, int(np.floor(points[:, 0].min())))
                y0 = max(0, int(np.floor(points[:, 1].min())))
                x1 = min(width, int(np.ceil(points[:, 0].max())))
                y1 = min(height, int(np.ceil(points[:, 1].max())))
                if x1 > x0 and y1 > y0:
                    boxes.append(OCRBox(str(text).strip(), max(0.0, min(1.0, confidence)), (x0, y0, x1 - x0, y1 - y0)))
            return boxes
