import numpy as np
import cv2
import os
from dataset import Dataset
from mvo import MVO

BACKGROUND_LIGHTNESS = 255
COLOR_GROUND_TRUTH = (232, 69, 51)
COLOR_ODOMETRY = (41, 41, 214)


def determine_plot_params(poses, poses_range=None):
    if poses_range:
        positions = poses[slice(*poses_range), :, 3]
    else:
        positions = poses[:, :, 3]

    xmin, xmax = np.min(positions[:, 0]), np.max(positions[:, 0])
    zmin, zmax = np.min(positions[:, 2]), np.max(positions[:, 2])
    extents = xmax - xmin, zmax - zmin
    plot_size = (500, 550)
    scale = plot_size[0] / extents[0], plot_size[1] / extents[1]
    origin = -xmin * scale[0], zmax * scale[1]

    return plot_size, origin, scale


def annotate(img):
    global COLOR_GROUND_TRUTH, COLOR_ODOMETRY

    img = cv2.putText(
        img,
        "Ground truth",
        (400, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_GROUND_TRUTH,
        1,
    )
    img = cv2.putText(
        img,
        "Estimated position",
        (400, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_ODOMETRY,
        1,
    )
    return img


def test(dataset, detector, plot=True, play_camera=False, output_file=None):
    # Lucas-Kanade flow parameters
    lk_params = dict(
        winSize=(21, 21),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    mvo = MVO(dataset, detector, lk_params)
    plot_size, origin, scale = determine_plot_params(dataset.poses, (0, 2000))
    trajectory = BACKGROUND_LIGHTNESS * np.ones(shape=plot_size + (3,), dtype=np.uint8)
    trajectory = annotate(trajectory)

    for i in range(1500):
        if not dataset.has_next():
            break

        mvo.process_frame()

        pos = mvo.abs_position()
        pos_est = mvo.position()

        x_est, _, z_est = [int(x) for x in pos_est]
        x, _, z = [int(x) for x in pos]

        # Plot ground truth trajectory
        trajectory = cv2.circle(
            trajectory,
            (int(origin[0] + scale[0] * x), int(origin[1] - scale[1] * z)),
            1,
            COLOR_GROUND_TRUTH,
            2,
        )
        # Plot trajectory estimated by odometry and scaled using ground truth data
        trajectory = cv2.circle(
            trajectory,
            (int(origin[0] + scale[0] * x_est), int(origin[1] - scale[1] * z_est)),
            1,
            COLOR_ODOMETRY,
            2,
        )

        if plot:
            cv2.imshow("trajectory", trajectory)
        if play_camera:
            cv2.imshow("camera", mvo.frame)
        if plot or play_camera:
            k = cv2.waitKey(1)
            if k == 27:  # Escape key
                break

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cv2.imwrite(output_file, trajectory)

    cv2.destroyAllWindows()
