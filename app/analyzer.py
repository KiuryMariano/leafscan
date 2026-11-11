import base64
import logging

import cv2
import numpy as np

from ultralytics import SAM

logger = logging.getLogger(__name__)

SAM_MODEL_NAME = "sam2.1_t.pt"
MAX_SIDE = 1280
MIN_LEAF_FRACTION = 0.005

_model = None


def _get_model() -> SAM:
    global _model
    if _model is None:
        _model = SAM(SAM_MODEL_NAME)
    return _model


def _vegetation_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    veg = ((h >= 35) & (h <= 90) & (s >= 60) & (v >= 40)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, kernel)
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, kernel)
    return veg


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8).copy()
    flood = m.copy()
    ff_mask = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    m[flood == 0] = 1
    return m.astype(bool)


def _sam_refine(bgr: np.ndarray, veg: np.ndarray) -> np.ndarray | None:
    components = cv2.connectedComponentsWithStats(veg)
    num, _, stats, centroids = components
    if num <= 1:
        return None
    img_area = bgr.shape[0] * bgr.shape[1]
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    leaf = np.zeros(bgr.shape[:2], dtype=bool)
    used = 0
    for idx in order[:3]:
        i = idx + 1
        if stats[i, cv2.CC_STAT_AREA] < img_area * MIN_LEAF_FRACTION:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        try:
            result = _get_model()(
                bgr,
                bboxes=[[x, y, x + w, y + h]],
                retina_masks=True,
                device="cpu",
                verbose=False,
            )[0]
        except Exception:
            logger.exception("falha no SAM2 para o componente %s", i)
            return None
        if result.masks is None:
            continue
        mask = result.masks.data[0].cpu().numpy() > 0.5
        leaf |= _fill_holes(mask)
        used += 1
    return leaf if used else None


def detect_leaf(bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Retorna (máscara da folha, sam_foi_usado)."""
    veg = _vegetation_mask(bgr)
    refined = _sam_refine(bgr, veg)
    if refined is not None and refined.sum() >= veg.sum() * 0.5:
        return refined, True
    return _fill_holes(veg > 0), False


def analyze(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    leaf, used_sam = detect_leaf(bgr)
    img_area = bgr.shape[0] * bgr.shape[1]
    leaf_px = int(leaf.sum())
    if leaf_px < img_area * MIN_LEAF_FRACTION:
        raise ValueError("nenhuma folha detectada na imagem")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, _ = cv2.split(hsv)
    healthy = leaf & (hue >= 35) & (hue <= 90) & (sat >= 60)
    diseased = leaf & ~healthy

    healthy_pct = 100.0 * int(healthy.sum()) / leaf_px
    diseased_pct = 100.0 * int(diseased.sum()) / leaf_px

    overlay = _render_overlay(bgr, leaf, healthy, diseased, healthy_pct, diseased_pct)
    ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("falha ao codificar imagem de saída")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "leaf_detected": True,
        "used_sam": used_sam,
        "leaf_coverage_pct": round(100.0 * leaf_px / img_area, 2),
        "healthy_pct": round(healthy_pct, 2),
        "diseased_pct": round(diseased_pct, 2),
        "overlay_image": "data:image/jpeg;base64," + b64,
    }


def _render_overlay(
    bgr: np.ndarray,
    leaf: np.ndarray,
    healthy: np.ndarray,
    diseased: np.ndarray,
    healthy_pct: float,
    diseased_pct: float,
) -> np.ndarray:
    base = bgr.astype(np.float32)

    red_img = base.copy()
    red_img[..., 2] = 255
    red_img[..., 0] *= 0.15
    red_img[..., 1] *= 0.15
    green_img = base.copy()
    green_img[..., 1] = 255

    out = base.copy()
    out[diseased] = cv2.addWeighted(base, 0.45, red_img, 0.55, 0)[diseased]
    out[healthy] = cv2.addWeighted(base, 0.55, green_img, 0.45, 0)[healthy]

    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(leaf.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)

    cv2.rectangle(out, (0, 0), (out.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(out, f"boa: {healthy_pct:.1f}%", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 255, 160), 2)
    cv2.putText(out, f"contaminada: {diseased_pct:.1f}%", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 255), 2)
    return out
