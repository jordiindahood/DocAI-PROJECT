import numpy as np
import cv2
from pathlib import Path

def read_image_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def write_image_png(path: Path, img_bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")

def parse_yolo_line(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    nums = list(map(float, parts[1:]))
    if len(nums) == 4:
        return ("bbox", cls, np.array(nums, dtype=np.float32))
    if len(nums) >= 6 and len(nums) % 2 == 0:
        return ("poly", cls, np.array(nums, dtype=np.float32).reshape(-1, 2))
    return None

def load_yolo_labels(label_path: Path):
    if not label_path.exists():
        return []
    items = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = parse_yolo_line(line)
        if parsed is not None:
            items.append(parsed)
    return items

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def bbox_xywh_to_corners(xywh, w, h):
    xc, yc, bw, bh = xywh
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

def corners_to_bbox_xywh(corners, w, h):
    xs = corners[:, 0]
    ys = corners[:, 1]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw < 2 or bh < 2:
        return None

    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    bw_n = bw / w
    bh_n = bh / h
    return np.array([xc, yc, bw_n, bh_n], dtype=np.float32)

def poly_norm_to_pixels(poly, w, h):
    p = poly.copy().astype(np.float32)
    p[:, 0] *= w
    p[:, 1] *= h
    return p

def poly_pixels_to_norm(poly, w, h):
    p = poly.copy().astype(np.float32)
    p[:, 0] /= w
    p[:, 1] /= h
    return clip01(p)

def apply_homography_points(points_xy, H):
    pts = points_xy.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out

def transform_labels(labels, H, w, h):
    out = []
    for kind, cls, data in labels:
        if kind == "bbox":
            corners = bbox_xywh_to_corners(data, w, h)
            warped = apply_homography_points(corners, H)
            xywh = corners_to_bbox_xywh(warped, w, h)
            if xywh is not None:
                out.append(("bbox", cls, xywh))
        elif kind == "poly":
            pix = poly_norm_to_pixels(data, w, h)
            warped = apply_homography_points(pix, H)
            warped[:, 0] = np.clip(warped[:, 0], 0, w - 1)
            warped[:, 1] = np.clip(warped[:, 1], 0, h - 1)
            nrm = poly_pixels_to_norm(warped, w, h)
            if nrm.shape[0] >= 3:
                out.append(("poly", cls, nrm))
    return out

def serialize_labels(labels):
    lines = []
    for kind, cls, data in labels:
        if kind == "bbox":
            xc, yc, bw, bh = data.tolist()
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        else:
            flat = data.reshape(-1).tolist()
            lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in flat))
    return "\n".join(lines) + ("\n" if lines else "")

def write_labels(path: Path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_labels(labels), encoding="utf-8")
