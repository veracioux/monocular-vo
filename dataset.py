import os
from enum import IntEnum
import numpy as np
import cv2


class Camera(IntEnum):
    Left = 1
    Right = 2


class Dataset:
    def __init__(self, dataset_dir: str, sequence_id: str):
        self.sequence_id = sequence_id
        self._state_index = 0
        self.dataset_dir = dataset_dir

        self.calib = Dataset._read_calib_data(
            os.path.join(dataset_dir, "calib", sequence_id, "calib.txt")
        )
        self.poses = Dataset._read_float_array(
            os.path.join(dataset_dir, "poses", sequence_id + ".txt")
        ).reshape((-1, 3, 4))
        self.times = Dataset._read_float_array(
            os.path.join(dataset_dir, "sequences", sequence_id, "times.txt")
        )

        self._cached_images = None

    def pose(self) -> np.array:
        """Get the current pose from the dataset"""
        return self.poses[self._state_index]

    def time(self) -> float:
        return self.times[self._state_index]

    def image(self, camera=Camera.Left) -> np.array:
        camera = int(camera)
        path_format = "{}/sequences/{}/image_{}/{ind}.png".format(
            self.dataset_dir,
            self.sequence_id,
            "{}",
            ind=str(self._state_index).zfill(6),
        )
        if self._cached_images:
            left, right = self._cached_images
        else:
            left = right = None
        if camera & Camera.Left and not left:
            path = path_format.format(0)
            if os.path.isfile(path):
                left = cv2.imread(path)
        if camera & Camera.Right and not right:
            path = path_format.format(1)
            if os.path.isfile(path):
                right = cv2.imread(path_format.format(1))

        self._cached_images = (left, right)

        if left is None:
            return right
        if right is None:
            return left
        return (left, right)

    def advance(self):
        """Advance the dataset one frame"""
        self._state_index += 1
        self._cached_images = None

    def has_next(self):
        return os.path.isfile(
            os.path.join(
                self.dataset_dir,
                "sequences",
                self.sequence_id,
                "image_0",
                str(self._state_index + 1).zfill(6) + ".png",
            )
        )

    # Helpers
    @staticmethod
    def _read_float_array(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return np.array([float(val) for val in f.read().strip().split()])

    @staticmethod
    def _read_calib_data(path):
        values = []

        with open(path, "r", encoding="utf-8") as f:
            values = [float(v) for v in f.readline().split()[1:]]

        return np.array(values).reshape((3, 4))
