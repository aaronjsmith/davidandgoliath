"""Slice 3x3 rock sheet into transparent PNG sprites."""
import os
import cv2
import numpy as np

SRC = "rock-stone-set-element-of-nature-and-mountains-vector.jpg"
OUT_DIR = "rocks"
COLS, ROWS = 3, 3
TOLERANCE = 245


def trim_transparent(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return rgba
    pad = 2
    x1, x2 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + pad + 1)
    y1, y2 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + pad + 1)
    return rgba[y1:y2, x1:x2]


def remove_white(img: np.ndarray, tolerance: int) -> np.ndarray:
    bgr = img.astype(np.int16)
    white = (
        (bgr[:, :, 0] >= tolerance)
        & (bgr[:, :, 1] >= tolerance)
        & (bgr[:, :, 2] >= tolerance)
    )
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[white, 3] = 0
    return rgba


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sheet = cv2.imread(SRC)
    h, w = sheet.shape[:2]
    cell_w, cell_h = w // COLS, h // ROWS

    for row in range(ROWS):
        for col in range(COLS):
            x1, y1 = col * cell_w, row * cell_h
            x2 = x1 + cell_w if col < COLS - 1 else w
            y2 = y1 + cell_h if row < ROWS - 1 else h
            cell = sheet[y1:y2, x1:x2]
            rgba = trim_transparent(remove_white(cell, TOLERANCE))
            idx = row * COLS + col
            path = os.path.join(OUT_DIR, f"rock_{idx}.png")
            cv2.imwrite(path, rgba)
            print(f"Saved {path} ({rgba.shape[1]}x{rgba.shape[0]})")


if __name__ == "__main__":
    main()
