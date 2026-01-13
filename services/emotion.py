"""Emotion service starter template.

This module provides a small, well-documented `EmotionService` class that
can be used as a starting point for integrating a real emotion model.
The implementation below is intentionally minimal and dependency-light so
it's easy to read and extend.
"""

from typing import Any, Dict, Optional, Tuple, List
import logging
import random

try:
    from PIL import Image
    import numpy as np
except Exception:
    Image = None
    np = None

EMOTIONS: List[str] = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "disgust",
    "fear",
]

logger = logging.getLogger(__name__)


class EmotionService:
    """Lightweight emotion service template.

    Replace `_load_model` and `predict` with real implementations when you
    integrate a trained model (PyTorch, TensorFlow, ONNX, etc.).
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load a model if available. Currently sets a dummy marker.

        Implement model-loading logic here and set `self.model` to the
        loaded object.
        """
        logger.info("EmotionService: no model loaded, using dummy predictor")
        self.model = "dummy"

    def _preprocess(self, image: Any) -> Any:
        """Convert common image inputs to a numpy-like array when possible.

        Accepts PIL `Image.Image`, numpy arrays, or other raw inputs and
        returns a best-effort array suitable for a model.
        """
        if Image is not None and isinstance(image, Image.Image):
            if np is not None:
                return np.array(image)
            return image
        if np is not None and isinstance(image, np.ndarray):
            return image
        return image

    def predict(self, image: Any) -> Dict[str, float]:
        """Return a probability distribution over `EMOTIONS`.

        This template returns a deterministic-ish dummy distribution. When
        integrating a real model, replace this with calls to the model's
        inference function and post-processing.
        """
        _ = self._preprocess(image)

        rand = random.Random()
        try:
            rand.seed(hash(image))
        except Exception:
            pass

        probs: Dict[str, float] = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
        chosen = rand.choice(EMOTIONS)
        probs[chosen] += 0.1
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        return probs

    def top_emotion(self, image: Any) -> Tuple[str, float]:
        """Return the top emotion and its probability."""
        probs = self.predict(image)
        label, score = max(probs.items(), key=lambda x: x[1])
        return label, score


__all__ = ["EmotionService", "EMOTIONS"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    svc = EmotionService()
    logger.info("Available emotions: %s", EMOTIONS)
    sample = None
    if Image is not None:
        try:
            sample = Image.new("RGB", (48, 48), "gray")
        except Exception:
            sample = None
    print("Top emotion (sample):", svc.top_emotion(sample))
