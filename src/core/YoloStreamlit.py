import os
import shutil
import sys

import cv2
sys.path.append(os.getcwd())  # NOQA

import time
import traceback
import streamlit as st

from ultralytics import YOLO
from PIL import Image
from src.core.Infer import _display_processed_frame
from src.core.ClassificationModels import MotorBikeModels


# Global variables to control the video status
is_video_playing = False


def stop_video():
    global is_video_playing
    is_video_playing = False


@st.cache_resource
def load_yolo_model(model_path: str) -> YOLO:
    """
    Load a pretrained YOLO model from the specified path.

    Parameters
    ----------
    model_path : str
        The path to the pretrained YOLO model.

    Returns
    -------
    YOLO
        The pretrained YOLO model.
    """
    return YOLO(model_path)


def infer_image(model, **kwargs):
    """
    Perform inference on an uploaded image.
    Args:
        model: The pretrained models. It can be YOLO models or classification models.
    """

    source_img = st.sidebar.file_uploader(
        label="Upload Image",
        type=("jpg", "jpeg", "png", 'bmp', 'webp'),
    )

    col1, col2 = st.columns(2)

    # Col 1: Show the uploaded images
    with col1:
        if source_img:
            # Load the image for later inference
            uploaded_image = Image.open(source_img)

            # Show the uploaded image
            st.header("Source Image")
            st.image(source_img, use_column_width=True)

    # TODO: Add a button to perform inference


def infer_video(model, **kwargs):
    pass


def __camera_classify(model, conf: float):
    pass


def infer_camera(model, **kwargs):
    global is_video_playing

    # If conf key exists, meaning we are using YOLO models
    conf = kwargs.get("conf", None)
    webcam_url = kwargs.get("webcam_url", None)

    if not webcam_url:
        st.error("Please enter a valid webcam URL")
        return

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Start Camera'):
            is_video_playing = True

    with col2:
        if st.button('Stop Camera'):
            stop_video()

    video_cap = cv2.VideoCapture(webcam_url)
    st_frame = st.empty()
    model_type = None

    # Init the classification model
    if str(model).find('yolo') != -1:
        classification_model = YOLO(model)
        model_type = 'yolo'
    else:
        model_name = os.path.basename(model).split('.')[0]
        classification_model = MotorBikeModels(
            model=model_name,
            weight=model
        )
        model_type = 'classification'

    while is_video_playing:
        ret, frame = video_cap.read()
        if ret:
            _display_processed_frame(
                model_type=model_type,
                classification_model=classification_model,
                frame=frame,
                conf=conf,
                st_frame=st_frame
            )
