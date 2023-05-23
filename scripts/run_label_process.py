import sys
import os

sys.path.append(os.getcwd())

from utils.process import (
    video_process,
    video_indoor_process,
)

from utils.labeling_process import label_data

from utils.utils import modify_path_for_indoor

import argparse

from typing import Dict


def run(conf: Dict) -> None:
    # Create the save path if it's not exists
    os.makedirs(conf["save_path"], exist_ok=True)

    label_data(conf)


def parse_args() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="./notebooks/videos/iran-test2-vehicle-counting-night-result.mp4",
        help="a path to the image ",
    )

    parser.add_argument(
        "--save_path",
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

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)
