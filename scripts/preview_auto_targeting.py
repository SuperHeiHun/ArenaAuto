"""Preview coordinate-free opponent targeting on a local image; never uses ADB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from arena_auto.config.manager import ConfigManager
from arena_auto.models.config import AppConfig
from arena_auto.recognition.auto_targeting import AutoTargeting
from arena_auto.recognition.ocr import create_ocr_provider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "debug" / "auto_targeting_preview.png")
    args = parser.parse_args()
    config = ConfigManager(args.config).load() if args.config else AppConfig()
    image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error("无法读取截图")
    targeting = AutoTargeting(create_ocr_provider(config.ocr), config)
    cards = targeting.scan(image)
    challenge = targeting.read_challenge(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", targeting.draw(image))
    if not ok:
        raise RuntimeError("无法编码预览图")
    args.output.write_bytes(encoded.tobytes())
    print(json.dumps({"cards": [
        {"id": c.opponent.id, "identity": c.identity, "power": c.opponent.power,
         "card": c.region, "click": c.opponent.click_position, "source": c.target_source}
        for c in cards
    ], "challenge_count": str(challenge) if challenge else None,
        "output": str(args.output), "device_actions": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
