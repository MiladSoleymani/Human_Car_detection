import sys
import os

sys.path.append(os.getcwd())

from utils.process import (
    video_process,
    video_outdoor_process,
)

from utils.utils import modify_path_for_indoor

import argparse

from typing import Dict


def run(conf: Dict) -> None:
    if conf["place"] == "outdoor":
        video_process(conf)
    elif conf["place"] == "indoor":
        video_process(conf)
        print(f"old path: {os.path.split(conf['video_save_path'])}")

        # Change directory name
        conf["video_save_path"] = modify_path_for_indoor(conf["video_save_path"])
        print(f"new path: {os.path.split(conf['video_save_path'])}")

        video_outdoor_process(conf)


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
        "--yolo_object",
        type=str,
        default="yolov8x.pt",
        help="setting yolo type that want to use",
    )

    parser.add_argument(
        "--yolo_face",
        type=str,
        default="models/files/yolov8n-face.pt",
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
