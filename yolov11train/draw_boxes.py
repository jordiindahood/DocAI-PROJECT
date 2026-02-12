import cv2
import os

# ── Configuration ────────────────────────────────────────────────────────────
label = "/mnt/disk/Holberton/DocAI-API/data/yolo-augmented-dataset/labels/train/0000_page0001_aug01.txt"   # Path to the YOLO .txt label file
image = "/mnt/disk/Holberton/DocAI-API/data/yolo-augmented-dataset/images/train/0000_page0001_aug01.png"   # Path to the corresponding image file
# ─────────────────────────────────────────────────────────────────────────────

# Class names matching data.yaml
CLASS_NAMES = {0: "text"}

# Box styling
BOX_COLOR = (0, 255, 0)       # Green
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)
FONT_THICKNESS = 1


def parse_yolo_label(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            boxes.append((class_id, x_center, y_center, w, h))
    return boxes


def draw_boxes(img, boxes):
    img_h, img_w = img.shape[:2]
    result = img.copy()

    for class_id, xc, yc, w, h in boxes:
        # Convert from normalised YOLO coords to pixel coords
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)

        cv2.rectangle(result, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Label text
        class_name = CLASS_NAMES.get(class_id, str(class_id))
        label_text = f"{class_name}"
        (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)

        # Background rectangle for the text
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(
            result, label_text,
            (x1 + 2, y1 - 4),
            FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA,
        )

    return result


def main():
    if not label or not image:
        print("ERROR: Please set the 'label' and 'image' variables at the top of the script.")
        return

    if not os.path.isfile(image):
        print(f"ERROR: Image not found: {image}")
        return
    if not os.path.isfile(label):
        print(f"ERROR: Label file not found: {label}")
        return

    img = cv2.imread(image)
    if img is None:
        print(f"ERROR: Could not read image: {image}")
        return

    boxes = parse_yolo_label(label)
    print(f"Loaded {len(boxes)} box(es) from {label}")

    result = draw_boxes(img, boxes)

    # Save output next to the original image
    base, ext = os.path.splitext(image)
    output_path = f"/mnt/disk/Holberton/DocAI-API/data/saves/test_boxes{ext}"
    cv2.imwrite(output_path, result)
    print(f"Saved result to: {output_path}")

if __name__ == "__main__":
    main()
