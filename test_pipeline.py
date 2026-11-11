import json

import cv2
import numpy as np

from app.analyzer import analyze


def make_synthetic_leaf(size=900):
    rng = np.random.default_rng(7)
    img = np.full((size, size, 3), (215, 218, 222), dtype=np.uint8)

    noise = rng.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    leaf = np.zeros((size, size), np.uint8)
    center = (size // 2, size // 2)
    for (dx, dy, ax, by, ang) in [
        (0, 40, 110, 300, 20), (30, 10, 100, 280, 40), (-25, -10, 105, 290, -15),
        (10, -35, 95, 260, 50), (-15, 30, 90, 250, -45), (0, 0, 120, 260, 5),
    ]:
        c = (center[0] + dx, center[1] + dy)
        cv2.ellipse(leaf, c, (ax, by), ang, 0, 360, 255, -1)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    texture = rng.normal(0, 9, leaf.shape).astype(np.int16)
    green = img.copy()
    green[..., 1] = np.clip(green[..., 1].astype(np.int16) + 70 + texture, 0, 255)
    green[..., 0] = np.clip(green[..., 0].astype(np.int16) - 10 + texture, 0, 255)
    green[..., 2] = np.clip(green[..., 2].astype(np.int16) - 55 + texture, 0, 255)

    img[leaf > 0] = green[leaf > 0]

    spots = 0
    attempts = 0
    while spots < 60 and attempts < 600:
        attempts += 1
        x, y = rng.integers(0, size, 2)
        if leaf[y, x] == 0:
            continue
        r = int(rng.integers(8, 26))
        kind = rng.random()
        if kind < 0.55:
            color = (int(rng.integers(30, 60)), int(rng.integers(70, 110)), int(rng.integers(90, 140)))
        elif kind < 0.85:
            color = (int(rng.integers(40, 90)), int(rng.integers(160, 210)), int(rng.integers(210, 250)))
        else:
            color = (int(rng.integers(20, 50)),) * 3
        patch = img.copy()
        cv2.circle(patch, (int(x), int(y)), r, color, -1)
        edge = (slice(max(0, y - r - 2), y + r + 2), slice(max(0, x - r - 2), x + r + 2))
        soft = cv2.GaussianBlur(patch[edge], (0, 0), 2.5)
        img[edge] = soft
        spots += 1

    return img, spots


if __name__ == "__main__":
    import base64

    img, spots = make_synthetic_leaf()
    cv2.imwrite("outputs/sintetica.jpg", img)
    result = analyze(img)
    jpg = base64.b64decode(result["overlay_image"].split(",", 1)[1])
    overlay = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if overlay is None:
        raise RuntimeError("overlay decodificado é inválido")
    cv2.imwrite("outputs/sintetica_overlay.jpg", overlay)
    print(json.dumps({k: v for k, v in result.items() if k != "overlay_image"}, indent=2, ensure_ascii=False))
    print(f"manchas desenhadas: {spots}")
