import os
import sys
sys.path.append(os.getcwd())  # NOQA

import time
import traceback
import streamlit as st

from ultralytics import YOLO
from PIL import Image


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


def infer_uploaded_image(model, **kwargs):
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
