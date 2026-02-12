#!/usr/bin/env python

import cv2

def draw_results(image_path, results):
    img = cv2.imread(image_path)

    for r in results:
        x0, y0, x1, y1 = r["bbox"]
        label = f'{r["text"]} ({r["ocr_conf"]:.2f})'

        cv2.rectangle(img, (x0, y0), (x1, y1), (0,255,0), 2)
        cv2.putText(
            img, label,
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0,255,0),
            1
        )

    cv2.imshow("DocAI Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
