#!/usr/bin/env python3
import cv2
import numpy as np

# 2.18
from Extractor import Extractor

W = 1920 // 2
H = 1080 // 2
F = 270
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
fe = Extractor(K)


def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches, pose = fe.extract(img)
    if pose is None:
        return
    print("Matches", pose)
    for pt1, pt2 in matches:
        u1, v1 = fe.denormalize(pt1)
        u2, v2 = fe.denormalize(pt2)
        # print(p.pt,"-->",u, v)
        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    cv2.imshow("image", img)
    cv2.waitKey(1)


if __name__ == "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break
