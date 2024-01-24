import os
import traceback
import streamlit as st

import src.configs.config as config
from src.utils import *
from src.core.YoloStreamlit import *
from src.utils.CopyUtils import copy_folder

from pathlib import Path

st.set_page_config(
    page_title="Interactive Interface for Motorcycle",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


class GUI():
    def __init__(self):
        self.task_type = None
        self.model_type = None
        self.model_path = None

        self._setup_sidebar()
        self._setup_main()

    def __model_config(self):
        st.sidebar.header("Model Configuration")

        # Choosing task type
        self.task_type = st.sidebar.selectbox(
            "Select Task",
            ["Detection", "Classify"],
            key="task_type"
        )

        # Render the available models based on the task type
        if self.task_type == "Detection":
            self.model_type = st.sidebar.selectbox(
                "Select Model",
                config.DETECTION_MODEL_LIST
            )
        elif self.task_type == "Classify":
            self.model_type = st.sidebar.selectbox(
                "Select Model",
                config.CLASSIFY_MODEL_LIST
            )
        else:
            st.error("Currently only 'Detection' function is implemented")

        # Render model path
        if self.model_type:
            self.model_path = Path(config.DETECTION_MODEL_DIR, str(self.model_type))

        # Loading pretrained DL model
        try:
            self.model = load_yolo_model(self.model_path)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            st.error(f"Unable to load model. Please check the specified path: {self.model_path}")

        # Confidence threshold
        self.confidence = float(st.sidebar.slider(
            "Select Model Confidence", 30, 100, 50)) / 100

        # Save toggle
        self.save_toggle = st.sidebar.checkbox(
            "Save",
            key="save_toggle"
        )

        if self.save_toggle:
            src = 'runs/detect'
            dst = './temp/'
            os.makedirs(dst, exist_ok=True)

            copy_folder(src, dst)
            self.save_toggle = False  # Reset the checkbox state after the operation

    def __image_video_config(self):
        st.sidebar.header("Image/Video Configuration")

        src_selectbox = st.sidebar.selectbox(
            "Select Source",
            config.SOURCES_LIST,
            key="src_selectbox"
        )

        if src_selectbox == config.SOURCES_LIST[0]:  # Image
            infer_image(self.model, conf=self.confidence)
        elif src_selectbox == config.SOURCES_LIST[1]:  # Video
            infer_video(self.model, conf=self.confidence)
        elif src_selectbox == config.SOURCES_LIST[2]:  # Camera
            webcam_url = st.sidebar.text_input("Enter Webcam URL")
            if webcam_url:
                infer_camera(self.model, webcam_url=webcam_url, conf=self.confidence)

    def _setup_sidebar(self):
        self.__model_config()
        self.__image_video_config()

    def _setup_main(self):
        st.title("Interface for Bộ Công An")


if __name__ == "__main__":
    gui = GUI()
