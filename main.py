#!/usr/bin/env python3
import cv2
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
#4.05
W, H = 1920 // 2,  1080 // 2
F = 270
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])

cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T
frames = []

def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(img, K)
    frames.append(frame)
    if len(frames) <= 1:
        return

    pts, Rt = match_frames(frames[-1], frames[-2])
    frames[-1].pose = np.dot(Rt, frames[-2].pose)

    pts4d = triangulate(frames[-1].pose, frames[-2].pose, pts[:,0], pts[:,1])

    #reject points without enough parallax 
    good_pts4d = np.abs(pts4d[:,3])>0.005
    pts4d = pts4d[good_pts4d]
    pts4d /= pts4d[:,3:] #homogenous to 3d coordinates

    #reject the points behind the camera
    good_pts4d = pts4d[:, 2] >0
    pts4d = pts4d[good_pts4d]

    for pt1, pt2 in pts:
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
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
