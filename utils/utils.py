import numpy as np
import pandas as pd

from supervision.tools.detections import Detections
from supervision.video.source import get_video_frames_generator
from yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch

from typing import List, Dict, Tuple
import uuid
import os
import json
from collections import defaultdict
from pprint import pprint

import zipfile

import cv2


# calculate car center by finding center of bbox
def calculate_car_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# calculate car center by finding center of bbox
def calculate_down_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


def log(info: Dict, type: str, log_save_path: str) -> None:
    # info = pd.DataFrame(data=info)
    name = type + str(uuid.uuid4())[:13] + ".json"
    # info.to_csv(os.path.join(log_save_path, name), index=False)

    with open(os.path.join(log_save_path, name), "w") as file:
        json.dump(info, file)


def find_best_region(yolo_detection: Tuple, mtcnn_detection: List):
    best_detection = []
    for bbox, _, class_id, tracker_id in yolo_detection:
        if class_id == 0:  # do the process on persons labels
            for demography in mtcnn_detection:
                region = demography["region"]
                face_center = (
                    int(region["x"] + region["w"] / 2),
                    int(region["y"] + region["h"] / 2),
                )
                if (
                    face_center[0] > bbox[0]
                    and face_center[0] < bbox[2]
                    and face_center[1] > bbox[1]
                    and face_center[1] < bbox[3]
                ):
                    best_detection.append(
                        {
                            "id": tracker_id,
                            "region": region,
                            "age": demography["age"],
                            "dominant_gender": demography["dominant_gender"],
                        }
                    )

    return best_detection


def modify_path_for_indoor(path: str):
    file_name, format = os.path.splitext(os.path.basename(path))
    return os.path.join(os.path.dirname(path), f"{file_name}_indoor{format}")


def extract_area_coordinates(json_path: str):
    # Read the JSON file
    with open(json_path, "r") as file:
        data = json.load(file)

    poly_coord = defaultdict(lambda: {"area": [], "distance": 0})

    # Extract the area coordinates
    for key, value in data.items():
        poly_coord[key]["distance"] = value["distance"]
        poly_coord[key]["area"].extend(
            [(coord["x"], coord["y"]) for coord in value["coord"]]
        )

    pprint(poly_coord)
    return poly_coord


def extract_line_coordinates(json_path: str):
    with open(json_path, "r") as file:
        data = json.load(file)

    return [
        [
            (value["start"]["x"], value["start"]["y"]),
            (value["end"]["x"], value["end"]["y"]),
        ]
        for key, value in data.items()
    ]


def combine_frame_with_heatmap(frame, heatmap, save_path: str):
    # Resize the heat map to match the frame size
    heat_map_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    # Combine the frame and heat map
    overlay = cv2.addWeighted(frame, 0.7, heat_map_resized, 0.5, 0)

    cv2.imwrite(os.path.join(save_path, "overlay_heatmap.jpg"), overlay)


def extract_folder_name(path: str):
    return os.path.splitext(os.path.basename(path))


def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
