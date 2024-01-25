from ultralytics import YOLO
import os
import sys
sys.path.append(os.getcwd())  # NOQA
import cv2
import time


def __predict_with_YOLO(model: str, original_img, bb_annotations: list):
    model = YOLO(model)


def __predict_with_classification(model, original_img, bb_annotations: list):
    pass


def _display_processed_frame(classification_model, frame, st_frame, conf: float = 0.5, **kwargs):
    image = cv2.resize(frame, (640, 480))

    # First, perform object detection on the frame
    model = YOLO(model='src/configs/weights/detection/yolov8m.pt')
    detection_result = model.predict(image, conf=conf, classes=3)

    orginal_img = image.copy()
    bounding_boxes = detection_result[0].boxes.xywh.numpy().tolist()

    # Second, perform classification on the bounding boxes
    if classification_model.lower().find('yolo') != -1:
        processed_frame = __predict_with_YOLO(classification_model, orginal_img, bounding_boxes)
    else:
        processed_frame = __predict_with_classification(classification_model, orginal_img, bounding_boxes)

    # Display the processed frame
    st_frame.image(processed_frame, use_column_width=True, caption='Detected Motorbikes')
