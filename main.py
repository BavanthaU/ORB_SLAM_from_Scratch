#!/usr/bin/env python3
import cv2
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
#4.05
W, H = 1920 // 2,  1080 // 2
F = 270
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])

cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
    
    def display_map(self):
        for f in self.frames:
            print(f.id)
            print(f.pose)
            print()
        # for p in points:
        #     print(p.xyz)



#classes
mapp = Map()

        
class Point(object):
    #3D point in world
    def __init__(self, mapp, loc):
        self.xyz = loc
        self.frames = []
        self.idxs = []

        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id <= 1:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    # homogenous 3D cordinates
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:,3:] #homogenous to 3d coordinates

    #reject points without enough parallax 
    # reject the points behind the camera
    good_pts4d = (np.abs(pts4d[:,3])>0.005)& (pts4d[:, 2] >0)
    
    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    cv2.imshow("image", img)
    cv2.waitKey(1)

    #3d world cordinates
    mapp.display_map()


if __name__ == "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break
