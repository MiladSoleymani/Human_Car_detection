import sys
import os

sys.path.append(os.getcwd())

from utils.process import (
    video_process,
)

import argparse

from typing import Dict


def run(conf: Dict) -> None:
    video_process(conf)


def parse_args() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_target_path",
        type=str,
        default="/home/szsoroush/Human_Car_detection/notebooks/iran-test2-vehicle-counting-night.mp4",
        help="target path",
    )

    parser.add_argument(
        "--video_save_path",
        type=str,
        default="./notebooks/videos/iran-test2-vehicle-counting-night-result.mp4",
        help="save path",
    )

    parser.add_argument(
        "--log_save_path",
        type=str,
        default="./log/person",
        help="save folder path for logs",
    )

    parser.add_argument(
        "--yolo_version",
        type=str,
        default="yolov8x.pt",
        help="setting yolo type that want to use",
    )

    parser.add_argument(
        "--place",
        type=str,
        default="outdoor",
        help="the place you want to run this program.",
        choices=["indoor", "outdoor"],
    )

    parser.add_argument(
        "--line_start",
        nargs="+",
        type=float,
        default=(200, 492),
        help="starting line's point that we count cars from that line. you can input this like : --line_start 20 30",
    )

    parser.add_argument(
        "--line_end",
        nargs="+",
        type=float,
        default=(1900, 492),
        help="ending line's point that we count cars from that line. you can input this like : --line_end 20 30",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)
