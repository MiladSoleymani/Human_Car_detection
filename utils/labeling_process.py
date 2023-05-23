import os
import cv2
from tqdm.notebook import tqdm
import shutil
import glob


from models.yolo import load_yolo
from utils.utils import extract_folder_name, zip_folder

from supervision.video.source import get_video_frames_generator
from supervision.video.dataclasses import VideoInfo


def label_on_video(conf):
    save_path = conf["save_path"]

    # create frame generator
    generator = get_video_frames_generator(conf["data_path"])
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(conf["data_path"])

    model, CLASS_NAMES_DICT, CLASS_ID = load_yolo(conf["yolo_object"])

    with open(os.path.join(save_path, "classes.txt"), "w") as file:
        for item in CLASS_NAMES_DICT.values():
            file.write(item + "\n")

    # loop over video frames
    for idx, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
        if idx == 3:
            break

        results = model(frame)
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        class_id = results[0].boxes.cls.cpu().numpy().astype(int)

        output_filename = os.path.join(save_path, f"frame_{idx}_.jpg")
        cv2.imwrite(output_filename, frame)

        # Create a .txt file for the current frame
        with open(os.path.join(save_path, f"frame_{idx}_.txt"), "w") as txt_file:
            txt_file.write(f"YOLO_OBB\n")
            for idx, data in enumerate(zip(xyxy, class_id)):
                bbox, output_class = data
                # Write the object's information to the .txt file in the format expected by labelimg
                if idx == (xyxy.shape[0] - 1):
                    txt_file.write(
                        f"0 {bbox[0]:0.6f} {bbox[1]:0.6f} {bbox[2]:0.6f} {bbox[3]:0.6f} -90.000000"
                    )
                else:
                    txt_file.write(
                        f"0 {bbox[0]:0.6f} {bbox[1]:0.6f} {bbox[2]:0.6f} {bbox[3]:0.6f} -90.000000\n"
                    )


def label_on_image(conf):
    save_path = conf["save_path"]

    model, CLASS_NAMES_DICT, CLASS_ID = load_yolo(conf["yolo_object"])

    with open(os.path.join(save_path, "classes.txt"), "w") as file:
        for item in CLASS_NAMES_DICT.values():
            file.write(item + "\n")

    results = model(conf["data_path"])
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    class_id = results[0].boxes.cls.cpu().numpy().astype(int)

    filename, _ = extract_folder_name(conf["data_path"])  # extract filename, extension
    shutil.copy(conf["data_path"], conf["save_path"])

    with open(os.path.join(save_path, f"{filename}.txt"), "w") as txt_file:
        txt_file.write(f"YOLO_OBB\n")
        for idx, data in enumerate(zip(xyxy, class_id)):
            bbox, output_class = data
            # Write the object's information to the .txt file in the format expected by labelimg
            if idx == (xyxy.shape[0] - 1):
                txt_file.write(
                    f"{output_class} {bbox[0]:0.6f} {bbox[1]:0.6f} {bbox[2]:0.6f} {bbox[3]:0.6f} -90.000000"
                )
            else:
                txt_file.write(
                    f"{output_class} {bbox[0]:0.6f} {bbox[1]:0.6f} {bbox[2]:0.6f} {bbox[3]:0.6f} -90.000000\n"
                )


def label_on_image_path(image_path: str, model_config_path: str, save_path: str):
    model, CLASS_NAMES_DICT, CLASS_ID = load_yolo(model_config_path)
    with open(os.path.join(save_path, "classes.txt"), "w") as file:
        for item in CLASS_NAMES_DICT.values():
            file.write(item + "\n")

    results = model(image_path)
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    class_id = results[0].boxes.cls.cpu().numpy().astype(int)

    filename, _ = extract_folder_name(image_path)  # extract filename, extension
    shutil.copy(image_path, save_path)

    with open(os.path.join(save_path, f"{filename}.txt"), "w") as txt_file:
        txt_file.write(f"YOLO_OBB\n")
        for idx, data in enumerate(zip(xyxy, class_id)):
            bbox, output_class = data
            # Write the object's information to the .txt file in the format expected by labelimg
            if idx == (xyxy.shape[0] - 1):
                txt_file.write(
                    f"0 {bbox[0]:0.6f} {bbox[1]:0.6f} {bbox[2]:0.6f} {bbox[3]:0.6f} -90.000000"
                )
            else:
                txt_file.write(
                    f"0 {bbox[0]:0.6f} {bbox[1]:0.6f} {bbox[2]:0.6f} {bbox[3]:0.6f} -90.000000\n"
                )


def label_data(conf):
    if os.path.isfile(conf["data_path"]):
        _, extension = extract_folder_name(conf["data_path"])

        if extension.lower() == ".mp4":
            print("The file is an MP4 video.")
            label_on_video(conf)
        elif extension.lower() in [".jpg", ".jpeg"]:
            print("The file is a JPEG image.")
            label_on_image(conf)
        else:
            print("The file has an unknown format.")
            label_on_image(conf)

    elif os.path.isdir(conf["data_path"]):
        folder_path = "/path/to/folder/*.{jpg,jpeg,png}"
        for file in folder_path:
            label_on_image_path(file, conf["yolo_object"], conf["save_path"])

    zip_folder(
        conf["save_path"],
        zip_path=os.path.join(
            os.path.dirname(conf["save_path"]),
            f"{os.path.basename(conf['save_path'])}.zip",
        ),
    )
