"""Shared synthetic-benchmark utilities for the Image Registration blog series.

Builds a synthetic "semiconductor inspection image" (die outlines, alignment crosses,
via/pad grid, periodic interconnect lines) so we have full control over ground truth:
every registration method below is scored against a transform (and, for the deformable
case, a displacement field) we know exactly, instead of eyeballing overlays.
"""
import numpy as np
import cv2

RNG_SEED = 0
SIZE = 256


def make_semiconductor_pattern(size: int = SIZE, seed: int = RNG_SEED) -> np.ndarray:
    """A synthetic wafer/die inspection image: outlines + pads + interconnect lines + noise."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 40, dtype=np.float32)

    # Die grid outlines (like scribe lines between dies on a wafer)
    step = size // 4
    for i in range(0, size + 1, step):
        cv2.line(img, (i, 0), (i, size), 160, 2)
        cv2.line(img, (0, i), (size, i), 160, 2)

    # Periodic fine interconnect lines inside each die cell
    for cy in range(step // 2, size, step):
        for cx in range(step // 2, size, step):
            for k in range(-3, 4):
                y = cy + k * 6
                if 0 <= y < size:
                    cv2.line(img, (cx - 30, y), (cx + 30, y), 100, 1)

    # Via/pad grid: small filled circles at regular intersections (alignment-friendly features)
    for cy in range(step // 2, size, step):
        for cx in range(step // 2, size, step):
            cv2.circle(img, (cx, cy), 10, 220, -1)
            cv2.circle(img, (cx, cy), 10, 60, 2)

    # Corner alignment marks (crosses), the kind a real overlay-metrology target uses
    mark_positions = [(20, 20), (size - 20, 20), (20, size - 20), (size - 20, size - 20)]
    for (mx, my) in mark_positions:
        cv2.line(img, (mx - 12, my), (mx + 12, my), 250, 2)
        cv2.line(img, (mx, my - 12), (mx, my + 12), 250, 2)

    # Sensor / photon noise
    noise = rng.normal(0, 6.0, size=img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    return img.astype(np.uint8)


def control_points(size: int = SIZE) -> np.ndarray:
    """Grid intersection points used as landmarks for quantitative registration error."""
    step = size // 4
    pts = [(cx, cy) for cy in range(0, size + 1, step) for cx in range(0, size + 1, step)]
    return np.array(pts, dtype=np.float32)


def affine_matrix(angle_deg: float, tx: float, ty: float, scale: float, center) -> np.ndarray:
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M.astype(np.float32)


def random_affine_matrix(rng: np.random.Generator, size: int = SIZE,
                          max_angle=6.0, max_trans=12.0, scale_range=(0.96, 1.04)) -> np.ndarray:
    angle = rng.uniform(-max_angle, max_angle)
    tx = rng.uniform(-max_trans, max_trans)
    ty = rng.uniform(-max_trans, max_trans)
    scale = rng.uniform(*scale_range)
    center = (size / 2, size / 2)
    return affine_matrix(angle, tx, ty, scale, center)


def elastic_field(size: int, rng: np.random.Generator, alpha: float = 8.0, sigma: float = 20.0):
    """Smooth random displacement field (thermal/vibration-style local drift), returned as
    (dx, dy) maps in pixels, plus the absolute sampling maps cv2.remap expects."""
    dx = rng.normal(0, 1, (size, size)).astype(np.float32)
    dy = rng.normal(0, 1, (size, size)).astype(np.float32)
    k = int(sigma * 3) | 1
    dx = cv2.GaussianBlur(dx, (k, k), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (k, k), sigma) * alpha

    grid_x, grid_y = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    map_x = grid_x + dx
    map_y = grid_y + dy
    return dx, dy, map_x, map_y


def warp_affine(img: np.ndarray, M: np.ndarray, size: int = SIZE) -> np.ndarray:
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=40)


def warp_points_affine(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.hstack([pts, ones])
    return (M @ homo.T).T


def invert_affine(M: np.ndarray) -> np.ndarray:
    M3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    return np.linalg.inv(M3)[:2, :]


def landmark_error(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1)))


def ecc_affine(ref: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Intensity-based affine registration via ECC. findTransformECC(template, input, ...)
    returns M mapping template(ref) -> input(moving) coords, the opposite direction from the
    moving -> ref convention used throughout this module, so it's inverted before returning."""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp = cv2.findTransformECC(ref.astype(np.float32), moving.astype(np.float32),
                                        warp, cv2.MOTION_AFFINE, criteria)
    except cv2.error as e:
        print("ECC failed to converge:", e)
    return invert_affine(warp)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))
