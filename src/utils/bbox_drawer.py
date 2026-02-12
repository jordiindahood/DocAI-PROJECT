import json
from pathlib import Path
import cv2

# -------- CONFIG --------
IMAGE_FILE = Path("data/processed/invoiceIMG/0002_page1.png")
ANN_FILE   = Path("data/annotations/invoiceANNOTATIONS/0002.json")
PAGE_IDX   = 0 # zero-based page index inside JSON for this image
# ------------------------

# Load image
img = cv2.imread(str(IMAGE_FILE))
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_FILE}")

# Load annotations
with open(ANN_FILE, "r") as f:
    data = json.load(f)

# Draw all bounding boxes for this page
for item in data[PAGE_IDX]:
    x0, y0, x1, y1 = item["bbox"]
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

# Show image
cv2.imshow("Annotated Boxes", img)
print("Press any key to close")
cv2.waitKey(0)
cv2.destroyAllWindows()
