#!/usr/bin/env python3

from dataset import Dataset, Camera
import cv2
import sys
from argparse import ArgumentParser
from test import test


def cli_callback(args):
    if not args.output and not args.plot and not args.camera:
        args.plot = True
    dataset = Dataset("./data", args.dataset)
    if args.detector == "FAST":
        detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    elif args.detector == "ORB":
        detector = cv2.ORB_create()
    elif args.detector == "SIFT":
        detector = cv2.SIFT_create()
    test(dataset, detector, args.plot, args.camera, args.output)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", "-d", default="00", help="dataset id (e.g. 00, 01, etc)"
    )
    parser.add_argument(
        "--detector",
        "-D",
        default="FAST",
        help="feature detector (FAST [default], ORB, SURF)",
    )
    parser.add_argument("--output", "-o", help="trajectory output file (PNG)")
    parser.add_argument(
        "--plot", "-p", default=False, action="store_true", help="plot the trajectory"
    )
    parser.add_argument(
        "--camera", "-c", default=False, action="store_true", help="show camera frames"
    )

    parser.set_defaults(func=cli_callback)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
