"""Verified optional language-model cache; no downloads at import time."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional
from urllib.request import urlopen
from zipfile import ZipFile

from arena_auto.exceptions import OCRProviderError

logger = logging.getLogger(__name__)

# Pinned upstream RapidOCR v3.10.0 model manifest; PP-OCRv6 already covers Japan.
KOREAN_MODELS = (
    ("ch_PP-OCRv5_det_mobile.onnx",
     "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
     "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"),
    ("korean_PP-OCRv5_rec_mobile.onnx",
     "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx",
     "cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b"),
)

# Official EasyOCR v1.3 recognition packs. The MD5 is over the extracted .pth,
# matching EasyOCR's own manifest; the ZIP member is selected by exact filename.
EASYOCR_MODELS = {
    "japan": ("japanese_g2.pth",
              "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/japanese_g2.zip",
              "bad5146990ccb1272cb0908440fbe15e"),
    "korean": ("korean_g2.pth",
               "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.zip",
               "befecf7b1ca2fffb5af814a51443682d"),
}

EASYOCR_DETECTION_MODEL = (
    "craft_mlt_25k.pth",
    "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
    "2f8227d2def4037cdb3b34389dcf9ec1",
)


def valid_model(path: Path, checksum: str, algorithm: str = "sha256") -> bool:
    if not path.is_file():
        return False
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == checksum.lower()


def download_model(path: Path, url: str, checksum: str, *, algorithm: str = "sha256",
                   archive_member: Optional[str] = None) -> Path:
    """Download official ONNX or a single known ZIP member, verify before replacing.

    Failed downloads never clobber an existing cache. ZIP paths are never used
    as filesystem paths, and TLS verification stays enabled.
    """
    if valid_model(path, checksum, algorithm):
        return path
    if not url.startswith("https://"):
        raise OCRProviderError("语言模型只允许 HTTPS 下载")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    logger.info("下载 OCR 语言模型: %s", path.name)
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False) as output:
            downloaded = Path(output.name)
            temporary.append(downloaded)
            with urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        candidate = downloaded
        if archive_member is not None:
            with ZipFile(downloaded) as archive, tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False
            ) as output:
                candidate = Path(output.name)
                temporary.append(candidate)
                info = archive.getinfo(archive_member)
                if info.is_dir() or info.file_size > 512 * 1024 * 1024:
                    raise OCRProviderError("模型 ZIP 成员无效或过大")
                with archive.open(info) as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        if not valid_model(candidate, checksum, algorithm):
            raise OCRProviderError(f"语言模型 {path.name} {algorithm.upper()} 校验失败")
        os.replace(candidate, path)
        logger.info("语言模型已安装并校验: %s", path)
        return path
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, OCRProviderError):
            raise
        raise OCRProviderError(f"语言模型 {path.name} 下载失败: {exc}") from exc
    finally:
        for part in temporary:
            part.unlink(missing_ok=True)
