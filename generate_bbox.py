import numpy as np 
import cv2 

# Higher thresholds focus on the most affected (strongest) areas of the heatmap
THRESHOLDS = [150, 180, 200]

def distance_transform(binary_img):

    dt = cv2.distanceTransform(
        255 - binary_img,
        cv2.DIST_L2,
        5
    )

    dt = cv2.normalize(
        dt,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return dt


def big_overlap(a, b):

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]

    thresh = max(area_a, area_b) * 0.5

    x_overlap = max(
        0,
        min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    )

    y_overlap = max(
        0,
        min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    )

    overlap_area = x_overlap * y_overlap

    return overlap_area > thresh


def merge_overlapping_boxes(boxes):

    merged = []

    while boxes:

        a = boxes.pop(0)

        keep = []

        for b in boxes:

            if not big_overlap(a, b):
                keep.append(b)

        merged.append(a)

        boxes = keep

    return merged


def get_bboxes_from_heatmap(cam_img):

    data_images = []

    # MULTI-THRESHOLD PROCESSING

    for th in THRESHOLDS:

        _, binary = cv2.threshold(
            cam_img,
            th,
            255,
            cv2.THRESH_BINARY
        )

        dt_img = distance_transform(binary)

        _, binary_dt = cv2.threshold(
            dt_img,
            10,
            255,
            cv2.THRESH_BINARY
        )

        data_images.append(binary_dt)

    # CONTOURS

    bboxes = []

    for img in data_images:

        contours, _ = cv2.findContours(
            img,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            x, y, w, h = cv2.boundingRect(cnt)

            x = max(0, x)
            y = max(0, y)

            w = max(0, min(w, cam_img.shape[1] - x))
            h = max(0, min(h, cam_img.shape[0] - y))

            if [x, y, w, h] != [0, 0, 224, 224]:
                bboxes.append([x, y, w, h])

    # MERGE + SORT

    bboxes = merge_overlapping_boxes(bboxes)

    bboxes.sort(
        key=lambda b: b[2] * b[3],
        reverse=True
    )

    # return highest rank bb (first one after sorting)
    #### Can change this to see diff resutls? 
    # print(bboxes)
    return bboxes[0]
    # print("Hello")
    # return [0,0,0,0]