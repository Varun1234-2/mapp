"""
Monocular metric depth estimation using Depth Anything V2 (Metric).

Model variants
--------------
* Indoor  : ``depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf``
* Outdoor : ``depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf``
* Fallback: ``depth-anything/Depth-Anything-V2-Small-hf`` (relative depth —
  scale must come entirely from triangulation).

All models run through HuggingFace *transformers*
``AutoModelForDepthEstimation`` and ``AutoImageProcessor``.  The metric
variants output depth in **metres**; the relative variant outputs an
inverse-depth map that is rescaled by the VO scale estimator.
"""

from __future__ import annotations

import time
import logging

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Model hub IDs — checked in order; the first that loads successfully wins.
_MODEL_IDS: dict[str, list[str]] = {
    "indoor": [
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "depth-anything/Depth-Anything-V2-Small-hf",
    ],
    "outdoor": [
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "depth-anything/Depth-Anything-V2-Small-hf",
    ],
}

# Sane depth clamps (metres)
_DEPTH_LIMITS: dict[str, tuple[float, float]] = {
    "indoor": (0.05, 10.0),
    "outdoor": (0.1, 80.0),
}


class DepthEstimator:
    """GPU-accelerated metric depth estimator.

    Parameters
    ----------
    env : ``"indoor"`` | ``"outdoor"``
    device : e.g. ``"cuda"`` or ``"cpu"``
    """

    def __init__(self, env: str = "indoor", device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.env = env
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.depth_min, self.depth_max = _DEPTH_LIMITS.get(env, (0.05, 80.0))

        candidates = _MODEL_IDS.get(env, _MODEL_IDS["indoor"])
        self.model = None
        self.processor = None
        self.model_id = ""
        self._is_metric = False

        for mid in candidates:
            try:
                logger.info("Trying depth model: %s", mid)
                self.processor = AutoImageProcessor.from_pretrained(mid)
                self.model = AutoModelForDepthEstimation.from_pretrained(mid)
                self.model.to(self.device).eval()
                self.model_id = mid
                self._is_metric = "metric" in mid.lower() or "Metric" in mid
                logger.info("Loaded depth model: %s (metric=%s)", mid, self._is_metric)
                break
            except Exception as exc:
                logger.warning("Could not load %s: %s", mid, exc)

        if self.model is None:
            raise RuntimeError(
                f"No depth model could be loaded for env={env!r}. "
                "Check your HuggingFace cache / network."
            )

    @property
    def is_metric(self) -> bool:
        """True if the loaded model outputs metric (metre) depth."""
        return self._is_metric

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run depth prediction.

        Args:
            image_bgr: H×W×3 uint8 BGR image.

        Returns:
            H×W float32 depth in metres (clamped to valid range).
            For non-metric models the values are relative — the caller
            must apply an external scale.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Depth model or processor is not loaded.")

        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]

        # Convert BGR → RGB PIL
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        predicted = getattr(outputs, "predicted_depth", None)
        if predicted is None:
            predicted = getattr(outputs, "logits", None)
        if predicted is None:
            raise RuntimeError("Depth model output missing 'predicted_depth' or 'logits'.")

        # Ensure 4-D tensor (batch, channel, height, width) for interpolation
        if predicted.ndim == 2:
            predicted = predicted.unsqueeze(0).unsqueeze(0)
        elif predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)

        # Resize to original resolution
        depth_tensor = torch.nn.functional.interpolate(
            predicted,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        depth = depth_tensor.detach().cpu().numpy().astype(np.float32)

        # Clamp
        depth = np.clip(depth, self.depth_min, self.depth_max)

        dt = time.perf_counter() - t0
        logger.debug("Depth inference: %.1f ms", dt * 1000)
        return depth

