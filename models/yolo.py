from ultralytics import YOLO

def load_yolo(yolo_version: str):
    model = YOLO(yolo_version)
    model.fuse()

    # dict maping class_id to class_name
    CLASS_NAMES_DICT = model.model.names
    # class_ids of interest - person, car, motorcycle, bus and truck
    CLASS_ID = [0, 2, 3, 5, 7]

    return model, CLASS_NAMES_DICT, CLASS_ID