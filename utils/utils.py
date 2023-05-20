import numpy as np
import pandas as pd

from supervision.tools.detections import Detections
from yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch

from typing import List, Dict, Tuple
import uuid
import os


# calculate car center by finding center of bbox
def calculate_car_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


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


def log(info: Dict, log_save_path: str) -> None:
    info = pd.DataFrame(data=info)
    name = "log_car_" + str(uuid.uuid4())[:13] + ".csv"
    info.to_csv(os.path.join(log_save_path, name), index=False)


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
    file_name, format = os.path.splitext(os.path.basename(path["video_save_path"]))
    return os.path.join(os.path.dirname(path, f"{file_name}_indoor{format}"))
