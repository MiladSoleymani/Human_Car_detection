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
    extract_multi_poly_coordinates,
    extract_multi_line_coordinates,
)

from typing import Dict


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

    try:
        lines = extract_line_coordinates(conf["line_path"])  # extract linee
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

    except Exception as e:
        print(f"the erro is {e = }")
        print("the config has been changed")

    multi_poly = extract_multi_poly_coordinates(conf["multi_poly"])
    multi_line = extract_multi_line_coordinates(conf["multi_line"])

    in_polygon = {}
    speed = {}
    multi_poly_log = defaultdict(
        lambda: {"tracker_ids": defaultdict(list), "object_count": defaultdict(int)}
    )
    multi_line_log = defaultdict(
        lambda: {"tracker_ids": defaultdict(list), "object_count": defaultdict(int)}
    )

    # open target video file
    with VideoSink(conf["video_save_path"], video_info) as sink:
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
                "eye_detected_count": 0,
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
                y_points = kpts[..., 0::3].astype(int)  # extract y points
                x_points = kpts[..., 1::3].astype(int)  # extract x points

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

                face_detections.tracker_id = np.array(face_tracker_id)

                mask = np.array(
                    [
                        tracker_id is not None
                        for tracker_id in face_detections.tracker_id
                    ],
                    dtype=bool,
                )
                face_detections.filter(mask=mask, inplace=True)

                for face_detection, x, y in zip(face_detections, x_points, y_points):
                    if (
                        x[0] >= (video_info.height - 5)
                        or x[1] >= (video_info.height - 5)
                        or y[1] >= (video_info.width - 5)
                        or y[0] >= (video_info.width - 5)
                    ):
                        continue

                    bbox, confidence, class_id, tracker_id = face_detection

                    center = ((x[0] + x[1]) // 2, (y[0] + y[1]) // 2)
                    log_eye_info[str(tracker_id)]["center_eye_loc"].append(
                        str((center[0], center[1]))
                    )

                    log_eye_info[str(tracker_id)]["eye_detected_count"] += 1

                    landmarks_heat_map[
                        int(center[0]) : int(center[0]) + 5,
                        int(center[1]) : int(center[1]) + 5,
                    ] += 1

                for detection_id in log_eye_info.keys():
                    if log_eye_info[str(detection_id)]["eye_time_eta"] == None:
                        log_eye_info[str(detection_id)]["eye_time_eta"] = (
                            log_eye_info[str(detection_id)]["eye_detected_count"]
                            / video_info.fps
                        )

                count = 0
                for detection_id in log_eye_info.keys():
                    if log_eye_info[str(detection_id)]["eye_time_eta"] != None:
                        count += 1

                if count > conf["log_save_steps"]:
                    log(log_eye_info, "log_eye_info_", conf["log_save_path"])

                    log_eye_info = defaultdict(
                        lambda: {
                            "eye_detected_count": 0,
                            "eye_time_eta": None,
                            "center_eye_loc": [],
                        }
                    )
                    # break

                if idx == (video_info.total_frames - 1):
                    log(log_eye_info, "log_eye_info_", conf["log_save_path"])
                    break

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

                    # print(f"try to calculate the intersection of different areas...")
                    # multi_poly_log
                for key, value in multi_poly.items():
                    result = cv2.pointPolygonTest(
                        np.array(value["area"], np.int32),
                        (int(cx), int(cy)),
                        False,
                    )

                    if result >= 0:
                        print(f"object {tracker_id} pass area {key}")
                        multi_poly_log[key]["tracker_ids"][
                            CLASS_NAMES_DICT[class_id]
                        ].append(int(tracker_id))

                        for detection_class in multi_poly_log[key][
                            "tracker_ids"
                        ].keys():
                            multi_poly_log[key]["object_count"][detection_class] = len(
                                set(multi_poly_log[key]["tracker_ids"][detection_class])
                            )

                for key, value in multi_line.items():
                    result = cv2.pointPolygonTest(
                        np.array(value["area"], np.int32),
                        (int(cx), int(cy)),
                        False,
                    )

                    if result >= 0:
                        print(f"object {tracker_id} pass area {key}")
                        multi_line_log[key]["tracker_ids"][
                            CLASS_NAMES_DICT[class_id]
                        ].append(int(tracker_id))

                        for detection_class in multi_line_log[key][
                            "tracker_ids"
                        ].keys():
                            multi_line_log[key]["object_count"][detection_class] = len(
                                set(multi_line_log[key]["tracker_ids"][detection_class])
                            )

                    # print(f"calculating is finished...")

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

            if len(log_info.keys()) > conf["log_save_steps"]:
                log(log_info, "person_car_", conf["log_save_path"])

                log_info = defaultdict(
                    lambda: {
                        "person_car": None,
                        "speed": None,
                        "car_type": None,
                        "location": [],
                    }
                )

            if idx == (video_info.total_frames - 1):
                log(log_info, "person_car_", conf["log_save_path"])
                break

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

            for key, value in multi_poly.items():
                cv2.polylines(
                    frame, [np.array(value["area"], np.int32)], True, (15, 228, 10), 3
                )

            for key, value in multi_line.items():
                cv2.polylines(
                    frame, [np.array(value["area"], np.int32)], True, (15, 228, 10), 3
                )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_color = (255, 255, 255)  # White color
            line_thickness = 2

            # Determine the size of the text
            text_size, _ = cv2.getTextSize(
                "Sample Text", font, font_scale, line_thickness
            )

            # Calculate the position of the text (centered in the frame)
            text_x = int((frame.shape[1] - text_size[0]) / 2)
            text_y = int((frame.shape[0] + text_size[1]) / 2)

            # Calculate the position of the list values
            list_text_x = int((frame.shape[1] - text_size[0]) / 2)
            list_text_y = (
                text_y - 30
            )  # Adjust the value to position the list above the frame

            # Iterate over the values and write them with spaces
            for key, value in multi_line_log.items():
                for sub_key, sub_value in value.items():
                    list_text = f"{sub_key}" + " : " + str(sub_value) + " "
                    list_text_size, _ = cv2.getTextSize(
                        list_text, font, font_scale, line_thickness
                    )
                    list_text_x += list_text_size[
                        0
                    ]  # Update the x-position for the next value
                    cv2.putText(
                        frame,
                        list_text,
                        (list_text_x, list_text_y),
                        font,
                        font_scale,
                        font_color,
                        line_thickness,
                    )

            # for value in line_counters.values():
            #     value["line_counter"].update(detections=detections)
            #     value["line_counter_annotator"].annotate(
            #         frame=frame, line_counter=value["line_counter"]
            #     )

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
    landmarks_heat_map = landmarks_heat_map / np.max(landmarks_heat_map)
    # Convert the heat map to color using a colormap
    heat_map_color = cv2.applyColorMap(
        (landmarks_heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    cv2.imwrite(
        os.path.join(conf["heatmap_savepath"], "heatmap_eyes.jpg"), heat_map_color
    )

    with open(os.path.join(conf["log_save_path"], "multi_poly_logs.json"), "w") as file:
        json.dump(multi_poly_log, file)

    with open(os.path.join(conf["log_save_path"], "multi_line_logs.json"), "w") as file:
        json.dump(multi_line_log, file)


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
            # face model prediction on single frame
            boxes, scores, class_ids, kpts, _ = face_model.detect(frame)
            face_xyxy = face_model.convert_xywh_to_xyxy(boxes)

            person_new_ids = []
            if boxes.size != 0:
                y_points = kpts[..., 0::3].astype(int)  # extract y points
                x_points = kpts[..., 1::3].astype(int)  # extract x points

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
                for face_detection, x, y in zip(face_detections, x_points, y_points):
                    if (
                        x[0] >= (video_info.height - 5)
                        or x[1] >= (video_info.height - 5)
                        or y[1] >= (video_info.width - 5)
                        or y[0] >= (video_info.width - 5)
                    ):
                        continue

                    _, _, _, tracker_id = face_detection

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

                    print("best_detections: ", best_detections)

                    for id in best_detections.keys():
                        log_info[str(id)]["age"] = best_detections[str(id)]["age"]
                        log_info[str(id)]["gender"] = best_detections[str(id)][
                            "dominant_gender"
                        ]

            for detection_id in log_info.keys():
                if detection_id not in detection_ids:
                    log_info[str(detection_id)]["eye_time_eta"] = (
                        log_info[str(detection_id)]["eye_detected_count"]
                        / video_info.fps
                    )

            count = 0
            for detection_id in log_info.keys():
                if log_info[str(detection_id)]["eye_time_eta"] != None:
                    count += 1

            if count > conf["log_save_steps"]:
                log(log_info, "indoor_", conf["log_save_path"])

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
                log(log_info, "indoor_", conf["log_save_path"])

            frame = face_model.draw_detections(
                frame, boxes, scores, kpts
            )  # change to eye

            sink.write_frame(frame)
