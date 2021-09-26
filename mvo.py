import cv2
import numpy as np
from dataset import Dataset


class MVO:
    def __init__(
        self,
        dataset: Dataset,
        detector,
        lk_params
    ):
        """
        lk_params {dict} -- Parameters for Lucas-Kanade optical flow
        detector {cv2.FeatureDetector} -- Feature detector
        """

        self.dataset = dataset
        self.detector = detector
        self.lk_params = lk_params
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.good_points = []

        self.frame = dataset.image()
        self.pose = dataset.pose()

        self.process_frame()

    def process_frame(self):
        """Process the next frame from the dataset."""

        self.old_frame = self.frame
        self.old_pose = self.pose

        self.dataset.advance()

        self.frame = self.dataset.image()
        self.pose = self.dataset.pose()

        if len(self.good_points) < 2000:
            self.keypoints = self._detect_keypoints(self.old_frame)

        # Calculate optical flow between frames, st holds status
        # of points from frame to frame
        self.p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self.old_frame, self.frame, self.keypoints, None, **self.lk_params
        )

        # Take the inliers
        self.good_points = self.p1[st == 1]
        old_good_points = self.keypoints[st == 1]

        E, mask = self._findEssentialMat(self.good_points, old_good_points)
        R, t = self._recoverPose(E, old_good_points, self.good_points)

        absolute_scale = np.linalg.norm(self.pose[:, 3] - self.old_pose[:, 3])
        self.t = self.t + absolute_scale * self.R @ t
        self.R = R @ self.R

    def position(self):
        return -self.t.flatten()

    def abs_position(self):
        """Absolute position estimated with the help of ground truth data"""
        return self.pose[:, 3]

    # Helper functions

    def _detect_keypoints(self, img):
        """Detect images and convert into numpy array"""

        keypoints = self.detector.detect(img)

        return np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(
            (-1, 1, 2)
        )

    def _findEssentialMat(self, old_points, new_points):
        E, mask = cv2.findEssentialMat(
            old_points,
            new_points,
            focal=self.dataset.calib[0, 0],
            pp=self.dataset.calib[0:2, 2],
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.9,
            mask=None,
        )
        return E, mask

    def _recoverPose(self, E, old_points, new_points):
        _, R, t, _ = cv2.recoverPose(
            E,
            old_points,
            new_points,
            R=None,
            t=None,
            focal=self.dataset.calib[0, 0],
            pp=self.dataset.calib[0:2, 2],
            mask=None,
        )
        return R, t
