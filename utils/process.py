import cv2
import numpy as np
from tqdm import tqdm

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.video.sink import VideoSink
from yolox.tracker.byte_tracker import BYTETracker
from dataclasses import dataclass

from deepface import DeepFace

from models.yolo import load_yolo, load_yolo_face, YOLOv8_face

from utils.utils import (
    detections2boxes,
    match_detections_with_tracks,
    calculate_car_center,
    find_best_region,
    log,
)

from typing import Tuple, Dict
import math
import time
import os


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def video_process(conf: Dict) -> Tuple[np.array, np.array]:
    start = time.time()
    # Load yolo pretrained model
    print("\nmodel summary : ", end="")
    model, CLASS_NAMES_DICT, CLASS_ID = load_yolo(conf["yolo_object"])
    face_model = YOLOv8_face(conf["yolo_face"])

    print(
        f"\nobject pretrained {conf['yolo_object'].replace('.pt', '')} classes : {CLASS_NAMES_DICT}"
    )
    detection_classe = {id: CLASS_NAMES_DICT[id] for id in CLASS_ID}
    print(f"\nour detection classe : {detection_classe}")

    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(conf["video_target_path"])
    print("\nvideo Info : ", end="")
    print(video_info)
    # create frame generator
    generator = get_video_frames_generator(conf["video_target_path"])
    # create LineCounter instance
    LINE_START = Point(conf["line_start"][0], conf["line_start"][1])
    LINE_END = Point(conf["line_end"][0], conf["line_end"][1])
    line_counter = LineCounter(start=LINE_START, end=LINE_END)
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(
        color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4
    )
    line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=0.4)

    # open target video file
    with VideoSink(conf["video_save_path"], video_info) as sink:
        log_info_person = {
            "id": [],
            "age": [],
            "gender": [],
        }

        log_info_car = {
            "id": [],
            "car_type": [],
            "speed": [],
        }

        detected_tracker_id = []
        # loop over video frames
        for idx, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
            if idx == 200:
                break

            # model prediction on single frame and conversion to supervision Detections
            boxes, scores, classids, kpts, eyes = face_model.detect(frame)
            frame = face_model.draw_detections(
                frame, boxes, scores, kpts
            )  # change to eye
            results = model(frame)

            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            )
            # filtering out detections with unwanted classes
            mask = np.array(
                [class_id in CLASS_ID for class_id in detections.class_id], dtype=bool
            )
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )
            tracker_id = match_detections_with_tracks(
                detections=detections, tracks=tracks
            )
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array(
                [tracker_id is not None for tracker_id in detections.tracker_id],
                dtype=bool,
            )
            detections.filter(mask=mask, inplace=True)

            for _, _, class_id, tracker_id in detections:
                if class_id == 0 and tracker_id not in log_info_person["id"]:
                    log_info_person["id"].append(tracker_id)
                    log_info_person["age"].append(None)
                    log_info_person["gender"].append(None)

                elif class_id != 0 and tracker_id not in log_info_car["id"]:
                    log_info_car["id"].append(tracker_id)
                    log_info_car["car_type"].append(None)
                    log_info_car["speed"].append(None)

            if len(log_info_person["id"]) > 5:
                log(log_info_person, conf["log_save_path"])

                log_info_person = {
                    "id": [],
                    "age": [],
                    "gender": [],
                }

            if len(log_info_car["id"]) > 5:
                log(log_info_car, conf["log_save_path"])

                log_info_car = {
                    "id": [],
                    "car_type": [],
                    "speed": [],
                }

            if idx == (video_info.total_frames - 1):
                log(log_info_car, conf["log_save_path"])
                log(log_info_person, conf["log_save_path"])
                break

            # format custom labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id in detections
            ]

            # updating line counter
            line_counter.update(detections=detections)
            # annotate and display frame
            frame = box_annotator.annotate(
                frame=frame, detections=detections, labels=labels
            )
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            sink.write_frame(frame)
