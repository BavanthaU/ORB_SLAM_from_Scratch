import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform



def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


class Extractor(object):
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    # project 3D points to 2D points
    def denormalize(self, pt):
        ret = np.dot(self.K, np.array((pt[0], pt[1], 1.0)))
        return int(round(ret[0])), int(round(ret[1]))

    # project 2D points to 3D points expressed w.r.t. to world coordinates
    def normalize(self, pts):
        # select index 0, 1 [0:2]--> [x, y, z] --> [x,y]
        # K^(-1) * (2D pixels) = 3D point expressed with respect to the world coordinate system
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def extractRT(self, E):
        W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        U, w, Vt = np.linalg.svd(E)
        assert np.linalg.det(U) > 0
        if np.linalg.det(Vt) > 0:
            Vt *= -1.0
        R = np.dot(np.dot(U, W), Vt)
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), Vt)
        t = U[:, 2]
        Rt = np.concatenate([R, t.reshape(3,1)], axis=1)
        return Rt

    def extract(self, img):
        # detecting
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        # matching
        ret = []
        if self.last is not None:
        # Lowe test
        # retaining only those where the distance of the best match is less than 0.75 times the distance of the second-best match. 
        # The coordinates of the matched keypoints from both images are then appended to the 'ret' list.
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # filter
        Rt = None
        if len(ret) > 0:
            ret = np.array(ret)
            print(ret.shape)
            # normalize coords: ( in the process of adding camera intrinsic properties)
            # 2D points to 3D points expressed w.r.t. to world coordinates
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])
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
                                    # FundamentalMatrixTransform,
                                    EssentialMatrixTransform,  # since camera is calibrated
                                    min_samples=8,
                                    # residual_threshold=1,
                                    residual_threshold=0.005,
                                    max_trials=200)

            ret = ret[inliers]

            Rt = self.extractRT(model.params)

        # return
        self.last = {'kps': kps, 'des': des}
        return ret, Rt
