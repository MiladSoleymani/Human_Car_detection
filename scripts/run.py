import sys
import os

sys.path.append(os.getcwd())

from utils.process import (
    video_process,
    video_indoor_process,
)

from utils.utils import modify_path_for_indoor

import argparse

from typing import Dict


def run(conf: Dict) -> None:
    # Create the save path if it's not exists
    os.makedirs(conf["heatmap_savepath"], exist_ok=True)

    if conf["place"] == "outdoor":
        video_process(conf)
    elif conf["place"] == "indoor":
        # video_process(conf)
        print(f"old path: {conf['video_save_path']}")

        # Change directory name
        conf["video_save_path"] = modify_path_for_indoor(conf["video_save_path"])
        print(f"new path: {conf['video_save_path']}")

        video_indoor_process(conf)


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
        "--area_path",
        type=str,
        default="configs/areas.json",
        help="determine a area to calculate speeds",
    )

    parser.add_argument(
        "--line_path",
        type=str,
        default="configs/lines.json",
        help="determine a line area",
    )

    parser.add_argument(
        "--heatmap_savepath",
        type=str,
        default=os.getcwd(),
        help="determine a line area",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)
