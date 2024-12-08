#!/usr/bin/env pyhton3
import cv2
import numpy as np
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

IRt= np.eye(4)

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def extractRt(E):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
  U,d,Vt = np.linalg.svd(E)
  assert np.linalg.det(U) > 0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]
  ret = np.eye(4)
  ret[:3,:3] = R
  ret[:3,3] = t
  return ret

def extract(img):
  orb = cv2.ORB_create()
  # detection
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

  # extraction
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
  kps, des = orb.compute(img, kps)

  # return pts and des
  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def normalize(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
  ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
  ret /= ret[2]
  return int(round(ret[0])), int(round(ret[1]))

def match_frames(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.des, f2.des, k=2)
  # Lowe's ratio test
  ret = []
  idx1, idx2 = [],[]
  for m,n in matches:
    if m.distance < 0.75*n.distance:
      idx1.append(m.queryIdx)
      idx2.append(m.trainIdx)

      p1 = f1.pts[m.queryIdx]
      p2 = f2.pts[m.trainIdx]
      ret.append((p1, p2))
  
  assert len(ret) >= 8
  ret = np.array(ret)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)
  print(ret.shape)

# Fundamental Matrix vs. Essential Matrix:
# ----------------------------------------
# Fundamental Matrix (F):
# - Encodes the epipolar geometry between two uncalibrated camera views.
# - Relates corresponding points in the two images (pixel coordinates).
# - Works without knowing the cameras' intrinsic parameters (e.g., focal length).
# - Equation: x2.T * F * x1 = 0, where x1 and x2 are image points in pixels.
# - Defines the epipolar lines in the second image for a point in the first.

# Essential Matrix (E):
# - Encodes the epipolar geometry for calibrated cameras.
# - Relates corresponding points in normalized image coordinates.
# - Requires knowledge of the cameras' intrinsic parameters.
# - Encodes the relative rotation (R) and translation (T) between cameras:
#   E = R * [T]_x (where [T]_x is the skew-symmetric matrix of translation vector T).
# - Equation: x2.T * E * x1 = 0, where x1 and x2 are normalized coordinates.

# Key Differences:
# - Fundamental matrix works with uncalibrated cameras, while the essential matrix assumes calibrated cameras.
# - Fundamental matrix uses raw pixel coordinates, while the essential matrix uses normalized coordinates.
# - The essential matrix explicitly encodes the relative motion (rotation and translation) between the two cameras.

  model, inliers = ransac((ret[:, 0], ret[:, 1]),
                          EssentialMatrixTransform,
                          #FundamentalMatrixTransform,
                          min_samples=8,
                          # residual_threshold=1,
                          residual_threshold=0.05,
                          max_trials=1000)

  # ignore outliers
  Rt = extractRt(model.params)

  # return
  return idx1[inliers], idx2[inliers], Rt

class Frame(object):
  def __init__(self, mapp, img, K):
    self.K = K
    self.Kinv = np.linalg.inv(self.K)
    self.pose = IRt

    pts, self.des = extract(img)
    self.pts = normalize(self.Kinv, pts)

    self.id = len(mapp.frames)
    mapp.frames.append(self)


