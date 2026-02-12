import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    if maxWidth < 10 or maxHeight < 10:
        return image

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def _a4_like(pts, tol=0.45):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if w <= 0 or h <= 0:
        return False
    ratio = max(w, h) / min(w, h)
    a4_ratio = 1.41421356237
    return abs(ratio - a4_ratio) <= tol


def _largest_quad_contour(mask, img_area, min_area_ratio=0.18):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_ratio:
            return None

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    return None


def _white_page_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s_thresh = 70
    v_thresh = 160

    mask = cv2.inRange(s, 0, s_thresh) & cv2.inRange(v, v_thresh, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def _fallback_edge_quad(image_bgr, img_area, min_area_ratio=0.18):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_ratio:
            return None

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    return None


def normalize_document(image_bgr, debug=False):
    img_h, img_w = image_bgr.shape[:2]
    img_area = float(img_h * img_w)

    mask = _white_page_mask(image_bgr)
    quad = _largest_quad_contour(mask, img_area, min_area_ratio=0.18)

    if quad is not None and _a4_like(quad, tol=0.55):
        if debug:
            print("Page detected via white-mask (A4-like) → warping")
        return four_point_transform(image_bgr, quad)

    quad2 = _fallback_edge_quad(image_bgr, img_area, min_area_ratio=0.18)
    if quad2 is not None and _a4_like(quad2, tol=0.65):
        if debug:
            print("Page detected via edge fallback (A4-like) → warping")
        return four_point_transform(image_bgr, quad2)

    if debug:
        print("Page not detected (or not A4-like) → returning original")
    return image_bgr


def normalize_document_debug(image_bgr):
    img_h, img_w = image_bgr.shape[:2]
    img_area = float(img_h * img_w)

    mask = _white_page_mask(image_bgr)
    quad = _largest_quad_contour(mask, img_area, min_area_ratio=0.18)

    if quad is not None and _a4_like(quad, tol=0.55):
        out = four_point_transform(image_bgr, quad)
        return out, {
            "page_detected": True,
            "method": "white_mask",
            "original_shape": (img_h, img_w),
            "normalized_shape": out.shape[:2],
        }

    quad2 = _fallback_edge_quad(image_bgr, img_area, min_area_ratio=0.18)
    if quad2 is not None and _a4_like(quad2, tol=0.65):
        out = four_point_transform(image_bgr, quad2)
        return out, {
            "page_detected": True,
            "method": "edges_fallback",
            "original_shape": (img_h, img_w),
            "normalized_shape": out.shape[:2],
        }

    return image_bgr, {
        "page_detected": False,
        "method": "none",
        "original_shape": (img_h, img_w),
        "normalized_shape": (img_h, img_w),
    }
