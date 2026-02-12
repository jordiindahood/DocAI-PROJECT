import numpy as np
import cv2

def salt_pepper_noise(img, prob=0.02):
    output = img.copy()
    sp_noise = np.zeros_like(img)
    if len(img.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = img.shape[2]
        if colorspace == 3:
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    sp_noise[probs < (prob / 2)] = black
    sp_noise[probs > 1 - (prob / 2)] = white
    return sp_noise, output

def random_illumination(img, strength_range=(0.15, 0.55), gamma_range=(0.85, 1.15)):
    h, w = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0

    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (x - w / 2) / (w / 2)
    y = (y - h / 2) / (h / 2)

    v = np.random.randn(2).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)

    g = v[0] * x + v[1] * y
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)

    strength = np.random.uniform(*strength_range)
    gamma = np.random.uniform(*gamma_range)

    mask = (1.0 - strength) + (2.0 * strength) * g
    mask = np.clip(mask, 0.0, 2.0)
    mask = np.power(mask, gamma)

    if img_f.ndim == 2:
        out = img_f * mask
    else:
        out = img_f * mask[..., None]

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)

def random_blur(img, k_choices=(0, 3, 5)):
    k = int(np.random.choice(k_choices))
    if k <= 1:
        return img
    return cv2.GaussianBlur(img, (k, k), 0)

def random_color_jitter_bgr(img, brightness=0.10, contrast=0.15):
    img_f = img.astype(np.float32)

    b = np.random.uniform(-brightness, brightness) * 255.0
    c = np.random.uniform(1.0 - contrast, 1.0 + contrast)

    out = img_f * c + b
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def random_perspective_homography(w, h, max_jitter=0.08):
    src = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    jx = max_jitter * w
    jy = max_jitter * h

    dst = src.copy()
    dst[0] += [np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)]
    dst[1] += [np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)]
    dst[2] += [np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)]
    dst[3] += [np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)]

    H = cv2.getPerspectiveTransform(src, dst)
    return H

def warp_perspective(img, H):
    h, w = img.shape[:2]
    out = cv2.warpPerspective(
        img, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return out

def apply_augmentations(img, H=None, sp_prob_range=(0.0, 0.02)):
    out = img

    if H is None:
        h, w = out.shape[:2]
        H = random_perspective_homography(w, h)

    out = warp_perspective(out, H)
    out = random_illumination(out)
    out = random_color_jitter_bgr(out)
    out = random_blur(out)

    sp_prob = np.random.uniform(*sp_prob_range)
    if sp_prob > 0:
        _, out = salt_pepper_noise(out, prob=sp_prob)

    return out, H
