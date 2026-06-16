"""Strip white backgrounds from character JPGs → transparent PNGs."""
import cv2
import numpy as np

MAX_DIM = 1400


def border_white_mask(white: np.ndarray, open_size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    opened = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)

    n, labels = cv2.connectedComponents(opened)
    border = set()
    for row in (labels[0], labels[-1]):
        border.update(row.tolist())
    for col in (labels[:, 0], labels[:, -1]):
        border.update(col.tolist())
    border.discard(0)

    bg = np.isin(labels, list(border))
    bg = cv2.dilate(bg.astype(np.uint8), kernel, iterations=1).astype(bool)
    return bg


def remove_white_region(img: np.ndarray, tolerance: int, open_size: int = 5) -> np.ndarray:
    bgr = img.astype(np.int16)
    white = (
        (bgr[:, :, 0] >= 255 - tolerance)
        & (bgr[:, :, 1] >= 255 - tolerance)
        & (bgr[:, :, 2] >= 255 - tolerance)
    ).astype(np.uint8)

    bg = border_white_mask(white, open_size)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = np.where(bg, 0, 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rgba[:, :, 3] = cv2.morphologyEx(rgba[:, :, 3], cv2.MORPH_CLOSE, kernel, iterations=1)
    return rgba


def process(path_in: str, path_out: str, tolerance: int, split_halves: bool = False) -> None:
    img = cv2.imread(path_in, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = min(1.0, MAX_DIM / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    if split_halves:
        mid = w // 2
        left = remove_white_region(img[:, :mid], tolerance, open_size=3)
        right = remove_white_region(img[:, mid:], tolerance, open_size=3)
        rgba = np.concatenate([left, right], axis=1)
    else:
        rgba = remove_white_region(img, tolerance, open_size=5)

    cv2.imwrite(path_out, rgba)
    print(f"Saved {path_out} ({w}x{h})")


process("lpjw_7adp_230906.jpg", "lpjw_7adp_230906.png", tolerance=18, split_halves=True)
process("v45d_bdsn_230906.jpg", "v45d_bdsn_230906.png", tolerance=28, split_halves=False)
