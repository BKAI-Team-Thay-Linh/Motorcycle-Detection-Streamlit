import os
import sys
from PIL import Image
sys.path.append(os.getcwd())  # NOQA

import cv2
import time

from src.utils.ImageUtils import crop_bb, draw_bb
from concurrent.futures import ProcessPoolExecutor, as_completed
from ultralytics import YOLO


def __predict_with_YOLO(model: YOLO, original_img, bb_annotations: list):
    pass


def __predict_with_classification(model, original_img, bb_annotations: list):
    # Next, crop the bounding boxes from the original image
    crop_images = crop_bb(original_img, bb_annotations)

    # Then, perform classification on the cropped images
    classes = []
    for crop_image in crop_images:
        classes.append(model.infer(crop_image))

    # Finally, draw the bounding boxes on the original image
    processed_frame = draw_bb(thickness=2, font_scale=0.5, original_image=original_img,
                              bb_boxes=bb_annotations, classes=classes)

    return processed_frame


def _display_processed_frame(model_type: str, classification_model, frame, st_frame, conf: float = 0.5, **kwargs):
    cropped_size = (720, int(720 * 9 / 16))
    image = cv2.resize(frame, cropped_size)

    # First, perform object detection on the frame
    model = YOLO(model='src/configs/weights/detection/yolov8m.pt')
    detection_result = model.predict(image, conf=conf, classes=3)

    orginal_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    bounding_boxes = detection_result[0].boxes.xywh.numpy().tolist()

    # Second, perform classification on the bounding boxes
    if model_type == 'yolo':
        processed_frame = __predict_with_YOLO(classification_model, orginal_img, bounding_boxes)
    else:
        processed_frame = __predict_with_classification(classification_model, orginal_img, bounding_boxes)

    # Display the processed frame
    st_frame.image(processed_frame, use_column_width=True, caption='Detected Motorbikes')
