"""Remove David and Goliath — fill with sampled sky/ground colours."""
import cv2
import numpy as np

SRC = "qao1_nr1y_230906.jpg"
OUT = "qao1_nr1y_230906_clean.jpg"

orig = cv2.imread(SRC)
h, w = orig.shape[:2]
result = orig.copy()

HORIZON = int(h * 0.52)

sky_color = np.median(orig[int(h * 0.08): int(h * 0.18), int(w * 0.78): int(w * 0.92)], axis=(0, 1))
ground_color = np.median(orig[int(h * 0.88): int(h * 0.97), int(w * 0.05): int(w * 0.20)], axis=(0, 1))

FILL_START = int(w * 0.26)

sky_layer = np.full((HORIZON, w - FILL_START, 3), sky_color, dtype=np.uint8)
for y in range(HORIZON):
    t = y / HORIZON
    sky_layer[y] = np.clip(sky_layer[y] * (1 - t * 0.06), 0, 255).astype(np.uint8)

ground_h = h - HORIZON
ground_layer = np.full((ground_h, w - FILL_START, 3), ground_color, dtype=np.uint8)
for y in range(ground_h):
    t = y / ground_h
    ground_layer[y] = np.clip(ground_layer[y] * (1 - t * 0.12), 0, 255).astype(np.uint8)

result[0:HORIZON, FILL_START:w] = sky_layer
result[HORIZON:h, FILL_START:w] = ground_layer

# Soft transition at the army/battlefield boundary
BLEND = 40
for i in range(BLEND):
    t = i / BLEND
    col = FILL_START - BLEND // 2 + i
    if 0 <= col < w:
        left_col = max(0, col - BLEND)
        result[:, col] = cv2.addWeighted(
            orig[:, left_col], 1 - t,
            result[:, col], t,
            0
        )

# Blend clouds softly from the original (no hard rectangular pastes)
cloud_src = orig[int(h * 0.04): int(h * 0.20), int(w * 0.74): int(w * 0.94)].astype(np.float32)
for cx_frac in (0.42, 0.62, 0.80):
    cw = int(w * 0.16)
    ch = int(h * 0.12)
    cloud = cv2.resize(cloud_src, (cw, ch))
    dx = int(w * cx_frac)
    dy = int(h * 0.05)
    x2, y2 = min(w, dx + cw), min(h, dy + ch)
    cw, ch = x2 - dx, y2 - dy
    patch = result[dy:y2, dx:x2].astype(np.float32)
    cloud = cloud[:ch, :cw]
    mask = (cloud.mean(axis=2) > 200).astype(np.float32) * 0.55
    mask = cv2.GaussianBlur(mask, (21, 21), 8)[..., np.newaxis]
    result[dy:y2, dx:x2] = np.clip(
        patch * (1 - mask) + cloud * mask, 0, 255
    ).astype(np.uint8)

# Draw small rocks directly (circles/ellipses — no square stamp artifacts)
rng = np.random.default_rng(42)
for _ in range(14):
    cx = int(rng.integers(FILL_START + 80, w - 60))
    cy = int(rng.integers(HORIZON + 40, h - 30))
    rx = int(rng.integers(12, 28))
    ry = int(rng.integers(8, 18))
    shade = tuple(int(c * rng.uniform(0.75, 0.92)) for c in ground_color)
    cv2.ellipse(result, (cx, cy), (rx, ry), rng.integers(0, 180), 0, 360, shade, -1)
    cv2.ellipse(result, (cx - 2, cy - 3), (max(4, rx // 3), max(3, ry // 3)), 0, 0, 360,
                tuple(min(255, int(c * 1.08)) for c in shade), -1)

cv2.imwrite(OUT, result, [cv2.IMWRITE_JPEG_QUALITY, 93])
print(f"Saved {OUT}")
