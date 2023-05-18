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

from models.yolo import (
    load_yolo, 
)

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

def video_car_process(conf: Dict) -> Tuple[np.array, np.array]:

    start = time.time()
    # Load yolo pretrained model
    print("\nmodel summary : ", end="")
    model, CLASS_NAMES_DICT, CLASS_ID = load_yolo(conf["yolo_version"])
    CLASS_ID = CLASS_ID[:4]
    
    print(f"\npretrained {conf['yolo_version'].replace('.pt', '')} classes : {CLASS_NAMES_DICT}")
    detection_classe = {id : CLASS_NAMES_DICT[id] for id in CLASS_ID}
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
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4)
    line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=0.4)

    # open target video file
    with VideoSink(conf["video_save_path"], video_info) as sink:
        
        log_info = {
            "id" : [],
            "car_type" : [],
            "speed" : [],
        }

        speeds = {}
        previous_positions = {}

        line1_start_calculate_speed = conf["line1_start_calculate_speed"]
        line1_end_calculate_speed = conf["line1_end_calculate_speed"]
        line2_start_calculate_speed = conf["line2_start_calculate_speed"]        
        line2_end_calculate_speed = conf["line2_end_calculate_speed"]

        # loop over video frames
        for idx, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
            
            # if idx == 50:
            #     break
            
            # model prediction on single frame and conversion to supervision Detections
            results = model(frame)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            for bbox, _, _, tracker_id in detections:
                speed = 0
                curr_pos = calculate_car_center(bbox)
                if previous_positions.get(tracker_id, None) is not None and curr_pos[1] >= line1_start_calculate_speed[1] and curr_pos[1] <= line2_start_calculate_speed[1]:
                    prev_pos = previous_positions.get(tracker_id, None)
                    distance_traveled = math.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)
                    time_between_frames = 1 / video_info.fps
                    speed = (distance_traveled ) / (time_between_frames)
                    speed /= 2
                    #print(f"#{tracker_id} speed : {speed} - coordinate: {curr_pos[1]}")
                
                if speeds.get(tracker_id, None) is None and speed != 0:
                    speeds[tracker_id] = [speed]
                elif speeds.get(tracker_id, None) is not None and speed != 0:
                    speeds[tracker_id].append(speed)
                previous_positions[tracker_id] = calculate_car_center(bbox)
            
            labels = []
            for _, confidence, class_id, tracker_id in detections:
                if tracker_id in speeds.keys():
                    # labels.append(f"{0.2*np.mean(speeds[tracker_id]):0.1f}km/h")
                    labels.append(f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f} {0.2*np.mean(speeds[tracker_id]):0.1f}km/h")
                else:
                    # labels.append("NA km/h")
                    labels.append(f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f} NA km/h")

            for label in labels:
                if label.split()[0] not in log_info["id"] and label.split()[3] != 'NA':
                    log_info["id"].append(label.split()[0])
                    log_info["car_type"].append(label.split()[1])
                    log_info["speed"].append(label.split()[2])

            if len(log_info["id"]) > 5:
                log(log_info, conf["log_save_path"])

                log_info = {
                    "id" : [],
                    "car_type" : [],
                    "speed" : [],
                }

            elif idx == (video_info.total_frames - 1):
                log(log_info, conf["log_save_path"])
                break 

            # updating line counter
            line_counter.update(detections=detections)
            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            cv2.line(frame, line1_start_calculate_speed, line1_end_calculate_speed, (255, 0, 0), thickness=2)
            cv2.line(frame, line2_start_calculate_speed, line2_end_calculate_speed, (255, 0, 0), thickness=2)
            sink.write_frame(frame)

    pass



def video_person_process(conf: Dict) -> Tuple[np.array, np.array]:

    start = time.time()
    # Load yolo pretrained model
    print("\nmodel summary : ", end="")
    model, CLASS_NAMES_DICT, CLASS_ID = load_yolo(conf["yolo_version"])
    CLASS_ID = [CLASS_ID[0]]

    print(f"\npretrained {conf['yolo_version'].replace('.pt', '')} classes : {CLASS_NAMES_DICT}")
    detection_classe = {id : CLASS_NAMES_DICT[id] for id in CLASS_ID}
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
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4)
    line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=0.4)

    # open target video file
    with VideoSink(conf["video_save_path"], video_info) as sink:
        
        log_info = {
            "id" : [],
            "age" : [],
            "gender" : [],
        }

        detected_tracker_id = []
        # loop over video frames
        for idx, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
            
            if idx == 200:
                break

            # model prediction on single frame and conversion to supervision Detections
            results = model(frame)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            
            person_new_ids = []
            for _, _, class_id, tracker_id in detections:
                if tracker_id not in detected_tracker_id and CLASS_NAMES_DICT[class_id] == 'person':
                    person_new_ids.append(tracker_id)
                    detected_tracker_id.append(tracker_id)

            if len(person_new_ids) > 0:

                skip_iter = False
                try:
                    demographies_mtcnn = DeepFace.analyze(img_path=frame, detector_backend='mtcnn', actions=('age', 'gender'))
                except:
                    skip_iter = True

                if not skip_iter:

                    best_detections = find_best_region(detections, demographies_mtcnn) # finding best faces' bboxes
                    for demography in best_detections:
                        if demography["id"] in person_new_ids:
                            region = demography["region"]
                            age = demography["age"]
                            dominant_gender = demography["dominant_gender"]
                            starting_point = (region["x"], region["y"])
                            ending_point = (int(region["x"] + region["w"]), int(region["y"] + region["h"]))

                            cv2.rectangle(frame,
                                        starting_point,
                                        ending_point,
                                        (255, 255, 255), 2)

                            cv2.putText(frame,
                                        f'gender:{dominant_gender} age:{age}',
                                        (starting_point[0], starting_point[1]-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)        

                    for detection in best_detections:
                        if detection["id"] not in log_info["id"]:
                            log_info["id"].append(detection["id"])
                            log_info["age"].append(detection["age"])
                            log_info["gender"].append(detection["dominant_gender"])

            if len(log_info["id"]) > 5:
                log(log_info, conf["log_save_path"])

                log_info = {
                    "id" : [],
                    "age" : [],
                    "gender" : [],
                }

            elif idx == (video_info.total_frames - 1):
                log(log_info, conf["log_save_path"])
                break 

            # format custom labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

            # updating line counter
            line_counter.update(detections=detections)
            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            sink.write_frame(frame)

    pass