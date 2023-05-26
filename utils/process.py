import cv2
import numpy as np
from tqdm import tqdm
import os
import json
from collections import defaultdict

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator

# from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.video.sink import VideoSink
from yolox.tracker.byte_tracker import BYTETracker
from dataclasses import dataclass

from deepface import DeepFace

from models.yolo import load_yolo, YOLOv8_face
from models.line_counter import LineCounter, LineCounterAnnotator

from utils.utils import (
    find_best_region,
    detections2boxes,
    match_detections_with_tracks,
    log,
    extract_area_coordinates,
    extract_line_coordinates,
    calculate_car_center,
    calculate_down_center,
    combine_frame_with_heatmap,
)

from typing import Tuple, Dict


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def video_process(conf: Dict) -> None:
    # Load yolo pretrained model
    print("\nmodel summary : ", end="")
    model, CLASS_NAMES_DICT, CLASS_ID = load_yolo(conf["yolo_object"])
    print(f"{CLASS_NAMES_DICT.keys()}")
    face_model = YOLOv8_face(conf["yolo_face"])

    print(
        f"\nobject pretrained {conf['yolo_object'].replace('.pt', '')} classes : {CLASS_NAMES_DICT}"
    )
    detection_classe = {id: CLASS_NAMES_DICT[id] for id in CLASS_ID}
    print(f"\nour detection classe : {detection_classe}")

    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    face_byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(conf["video_target_path"])
    print("\nvideo Info : ", end="")
    print(video_info)
    # create frame generator
    generator = get_video_frames_generator(conf["video_target_path"])
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(
        color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4
    )

    poly_coord = extract_area_coordinates(
        conf["area_path"]
    )  # extract area and distance

    lines = extract_line_coordinates(conf["line_path"])  # extract line

    print(f"{len(lines) = }")

    line_counters = {
        i: {
            "line_counter": LineCounter(
                start=Point(x=lines[i][0][0], y=lines[i][0][1]),
                end=Point(x=lines[i][1][0], y=lines[i][1][1]),
            ),
            "line_counter_annotator": LineCounterAnnotator(
                thickness=1, text_thickness=1, text_scale=0.4
            ),
        }
        for i in range(len(lines))
    }
    in_polygon = {}
    speed = {}
    # open target video file
    with VideoSink(conf["video_save_path"], video_info) as sink:
        landmarks_time_map = np.zeros((video_info.height, video_info.width))
        landmarks_heat_map = np.zeros((video_info.height, video_info.width))
        heat_map = np.zeros((video_info.height, video_info.width), dtype=np.float32)

        log_info = defaultdict(
            lambda: {
                "person_car": None,
                "speed": None,
                "car_type": None,
                "location": [],
            }
        )

        log_eye_info = defaultdict(
            lambda: {
                "eye_detected_count": None,
                "eye_time_eta": None,
                "center_eye_loc": [],
            }
        )
        print(f"{video_info.total_frames = }")
        # loop over video frames
        for idx, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
            if idx == 500:
                break

            # face model prediction on single frame
            boxes, scores, class_ids, kpts, _ = face_model.detect(frame)
            face_xyxy = face_model.convert_xywh_to_xyxy(boxes)

            if boxes.size != 0:  # check if sth is detected or not
                x_points = kpts[..., 0::3].astype(int)  # extract x points
                y_points = kpts[..., 1::3].astype(int)  # extract y points

                print("\ntracking the faces")
                face_detections = Detections(
                    xyxy=face_xyxy,
                    confidence=scores,
                    class_id=class_ids.astype(int),
                )

                face_tracks = face_byte_tracker.update(
                    output_results=detections2boxes(detections=face_detections),
                    img_info=frame.shape,
                    img_size=frame.shape,
                )

                face_tracker_id = match_detections_with_tracks(
                    detections=face_detections, tracks=face_tracks
                )

                print(f"\n{face_tracker_id = }")
                face_detections.tracker_id = np.array(face_tracker_id)

                mask = np.array(
                    [
                        tracker_id is not None
                        for tracker_id in face_detections.tracker_id
                    ],
                    dtype=bool,
                )
                face_detections.filter(mask=mask, inplace=True)

                detection_ids = []
                for face_detections, x, y in zip(face_detections, x_points, y_points):
                    bbox, confidence, class_id, tracker_id = face_detections

                    center = ((x[0] + x[1]) // 2, (y[0] + y[1]) // 2)
                    log_info[str(tracker_id)]["center_eye_loc"].append(
                        str((center[0], center[1]))
                    )

                    log_eye_info[str(tracker_id)]["eye_detected_count"] += 1

                    landmarks_time_map[x[0], y[0]] += 1  # right eye
                    landmarks_time_map[x[1], y[1]] += 1  # left eye
                    landmarks_heat_map[x[0] : x[0] + 2, y[0] : y[0] + 2] += 1
                    landmarks_heat_map[x[1] : x[1] + 2, y[1] : y[1] + 2] += 1
                    detection_ids.append(str(tracker_id))

                for detection_id in log_eye_info.keys():
                    if detection_id not in detection_ids:
                        log_eye_info[str(tracker_id)]["eye_time_eta"] = (
                            log_eye_info[str(tracker_id)]["eye_detected_count"]
                            / video_info.fps
                        )

                count = 0
                for detection_id in log_eye_info.keys():
                    if log_eye_info[str(tracker_id)]["eye_time_eta"] != None:
                        count += 1

                if count > 5:
                    log(log_eye_info, "log_eye_info", conf["log_save_path"])

                    log_eye_info = defaultdict(
                        lambda: {
                            "eye_detected_count": None,
                            "eye_time_eta": None,
                            "center_eye_loc": [],
                        }
                    )
                    # break

                if idx == (video_info.total_frames - 1):
                    log(log_eye_info, "log_eye_info", conf["log_save_path"])
                    # break

            # object model prediction on single frame
            results = model(frame)

            xywh = results[0].boxes.xywh.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for confidence, cordinate in zip(confidences, xywh):
                if confidence > 0.5:
                    # Accumulate the confidence score in the heat map
                    (centerX, centerY, bbox_width, bbox_height) = cordinate.astype(
                        "int"
                    )
                    heat_map[
                        centerY
                        + int(bbox_height / 2)
                        - 10 : centerY
                        + int(bbox_height / 2),
                        centerX - int(bbox_width / 2) : centerX + int(bbox_width / 2),
                    ] += 1

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

            for bbox, confidence, class_id, tracker_id in detections:
                if class_id != 0:
                    cx, cy = calculate_car_center(bbox)

                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                    for key, value in poly_coord.items():
                        result = cv2.pointPolygonTest(
                            np.array(value["area"], np.int32),
                            (int(cx), int(cy)),
                            False,
                        )

                        if result >= 0:
                            if str(tracker_id) in in_polygon.keys():
                                in_polygon[str(tracker_id)] += 1
                            else:
                                in_polygon[str(tracker_id)] = 1
                        elif result < 0:
                            if str(tracker_id) in in_polygon.keys():
                                time = in_polygon[str(tracker_id)] / video_info.fps
                                speed[str(tracker_id)] = (
                                    value["distance"] / time
                                ) * 3.6

            for bbox, _, class_id, tracker_id in detections:
                if str(tracker_id) not in log_info.keys():
                    if class_id == 0:
                        log_info[str(tracker_id)]["person_car"] = "person"
                    else:
                        log_info[str(tracker_id)]["person_car"] = "car"

                        log_info[str(tracker_id)]["car_type"] = CLASS_NAMES_DICT[
                            class_id
                        ]

                    log_info[str(tracker_id)]["location"].append(
                        str(calculate_down_center(bbox))
                    )
                else:
                    log_info[str(tracker_id)]["location"].append(
                        str(calculate_down_center(bbox))
                    )

                    if class_id != 0 and str(tracker_id) in speed.keys():
                        log_info[str(tracker_id)]["speed"] = str(speed[str(tracker_id)])

            if len(log_info.keys()) > 5:
                log(log_info, "person_car", conf["log_save_path"])

                log_info = defaultdict(
                    lambda: {
                        "person_car": None,
                        "speed": None,
                        "car_type": None,
                        "location": [],
                    }
                )
                # break

            if idx == (video_info.total_frames - 1):
                log(log_info, "person_car", conf["log_save_path"])
                # break

            # format custom labels
            labels = []
            for bbox, confidence, class_id, tracker_id in detections:
                if class_id == 0:
                    labels.append(
                        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                    )
                else:
                    if str(tracker_id) in speed.keys():
                        speed_tracker_id = speed[str(tracker_id)]
                        labels.append(
                            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {speed_tracker_id:0.2f}km/h"
                        )
                    else:
                        labels.append(f"#{tracker_id} {CLASS_NAMES_DICT[class_id]}")

            for key, value in poly_coord.items():
                cv2.polylines(
                    frame, [np.array(value["area"], np.int32)], True, (15, 228, 10), 3
                )

            for value in line_counters.values():
                value["line_counter"].update(detections=detections)
                value["line_counter_annotator"].annotate(
                    frame=frame, line_counter=value["line_counter"]
                )

            # annotate and display frame
            frame = box_annotator.annotate(
                frame=frame, detections=detections, labels=labels
            )

            frame = face_model.draw_detections(
                frame, boxes, scores, kpts
            )  # change to eye

            sink.write_frame(frame)
    # Normalize the heat map
    heat_map = heat_map / np.max(heat_map)

    # Convert the heat map to color using a colormap
    heat_map_color = cv2.applyColorMap(
        (heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    cv2.imwrite(
        os.path.join(conf["heatmap_savepath"], "heatmap_cars.jpg"), heat_map_color
    )

    # save heatmap on videos
    combine_frame_with_heatmap(
        frame=frame, heatmap=heat_map_color, save_path=conf["heatmap_savepath"]
    )

    # Normalize the heat map
    landmarks_heat_map = landmarks_heat_map.T / np.max(landmarks_heat_map)
    # Convert the heat map to color using a colormap
    heat_map_color = cv2.applyColorMap(
        (landmarks_heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    cv2.imwrite(
        os.path.join(conf["heatmap_savepath"], "heatmap_eyes.jpg"), heat_map_color
    )

    print(f"time elapse: {np.sum(landmarks_time_map) / (2 * video_info.fps)}")


def video_indoor_process(conf: Dict) -> None:
    # Load yolo pretrained model
    print("\nmodel summary : ", end="")
    face_model = YOLOv8_face(conf["yolo_face"])

    print(f"\npretrained {conf['yolo_face'].replace('.pt', '')}")

    # create BYTETracker instance
    face_byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(conf["video_target_path"])
    print("\nvideo Info : ", end="")
    print(video_info)
    # create frame generator
    generator = get_video_frames_generator(conf["video_target_path"])
    # open target video file
    with VideoSink(conf["video_save_path"], video_info) as sink:
        log_info = defaultdict(
            lambda: {
                "age": None,
                "gender": None,
                "eye_detected_count": None,
                "eye_time_eta": None,
                "center_eye_loc": [],
                "eye_detected_count": 0,
            }
        )

        detected_tracker_id = []
        # loop over video frames
        for idx, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
            if idx == 400:
                break

            # face model prediction on single frame
            boxes, scores, class_ids, kpts, _ = face_model.detect(frame)
            face_xyxy = face_model.convert_xywh_to_xyxy(boxes)

            person_new_ids = []
            if boxes.size != 0:
                x_points = kpts[..., 0::3].astype(int)  # extract x points
                y_points = kpts[..., 1::3].astype(int)  # extract y points

                print("\ntracking the faces")
                face_detections = Detections(
                    xyxy=face_xyxy,
                    confidence=scores,
                    class_id=class_ids.astype(int),
                )

                face_tracks = face_byte_tracker.update(
                    output_results=detections2boxes(detections=face_detections),
                    img_info=frame.shape,
                    img_size=frame.shape,
                )

                face_tracker_id = match_detections_with_tracks(
                    detections=face_detections, tracks=face_tracks
                )

                print(f"\n{face_tracker_id = }")
                face_detections.tracker_id = np.array(face_tracker_id)

                mask = np.array(
                    [
                        tracker_id is not None
                        for tracker_id in face_detections.tracker_id
                    ],
                    dtype=bool,
                )
                face_detections.filter(mask=mask, inplace=True)

                detection_ids = []
                for face_detections, x, y in zip(face_detections, x_points, y_points):
                    _, _, _, tracker_id = face_detections

                    if tracker_id not in detected_tracker_id:
                        person_new_ids.append(tracker_id)
                        detected_tracker_id.append(tracker_id)

                    center = ((x[0] + x[1]) // 2, (y[0] + y[1]) // 2)
                    log_info[str(tracker_id)]["center_eye_loc"].append(
                        str((center[0], center[1]))
                    )
                    log_info[str(tracker_id)]["eye_detected_count"] += 1

                    detection_ids.append(str(tracker_id))

            if len(person_new_ids) > 0:
                skip_iter = False
                try:
                    demographies_mtcnn = DeepFace.analyze(
                        img_path=frame,
                        detector_backend="mtcnn",
                        actions=("age", "gender"),
                    )
                except:
                    skip_iter = True

                if not skip_iter:
                    best_detections = find_best_region(
                        face_detections, demographies_mtcnn
                    )  # finding best faces' bboxes

                    for detection in best_detections:
                        log_info[str(detection["id"])]["age"] = detection["id"]["age"]
                        log_info[str(detection["id"])]["gender"] = detection["id"][
                            "gender"
                        ]

            for detection_id in log_info.keys():
                if detection_id not in detection_ids:
                    log_info[str(tracker_id)]["eye_time_eta"] = (
                        log_info[str(tracker_id)]["eye_detected_count"] / video_info.fps
                    )

            count = 0
            for detection_id in log_info.keys():
                if log_info[str(tracker_id)]["eye_time_eta"] != None:
                    count += 1

            if count > 5:
                log(log_info, "indoor", conf["log_save_path"])

                log_info = defaultdict(
                    lambda: {
                        "age": None,
                        "gender": None,
                        "eye_detected_count": None,
                        "eye_time_eta": None,
                        "center_eye_loc": [],
                        "eye_detected_count": 0,
                    }
                )

            elif idx == (video_info.total_frames - 1):
                log(log_info, "indoor", conf["log_save_path"])
                break

            frame = face_model.draw_detections(
                frame, boxes, scores, kpts
            )  # change to eye

            sink.write_frame(frame)
