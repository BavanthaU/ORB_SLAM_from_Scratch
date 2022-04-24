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

    # def denormalize(self, pt, shape):
    #     return int(round(pt[0] + shape[0] // 2)), int(round(pt[1] + shape[1] // 2))

    # project 3D points to 2D points
    def denormalize(self, pt):
        ret = np.dot(self.K, np.array((pt[0], pt[1], 1.0)))
        return int(round(ret[0])), int(round(ret[1]))

    # project 2D points to 3D points expressed w.r.t. to world coordinates
    def normalize(self, pts):
        # select index 0, 1 [0:2]--> [x, y, z] --> [x,y]
        # K^(-1) * (2D pixels) = 3D point expressed with respect to the world coordinate system
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def extract(self, img):
        # detecting
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        # matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # filter
        if len(ret) > 0:
            ret = np.array(ret)
            print(ret.shape)
            # normalize coords: ( in the process of adding camera intrinsic properties)
            # 2D points to 3D points expressed w.r.t. to world coordinates
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)
            s, v, d = np.linalg.svd(model.params)
            print(v)
            ret = ret[inliers]

        # return
        self.last = {'kps': kps, 'des': des}
        return ret
