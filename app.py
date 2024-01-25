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

        # Choose the classification model
        choose_cls_model = st.sidebar.selectbox(
            "Select Classification Model",
            config.CLASSIFY_MODEL_LIST,
            key="model_selectbox"
        )

        self.model = Path(config.CLASSIFY_MODEL_DIR, str(choose_cls_model))
        print(f"==>> self.model: {self.model}")

        # Confidence threshold
        self.confidence = float(st.sidebar.slider(
            "Select Model Confidence", 30, 100, 40)) / 100

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
            webcam_url = st.sidebar.text_input(
                "Enter Webcam URL", value='rtsp://Cam2:Etcop2@2023Ai2@Cam26hc.cameraddns.net:556/Streaming/Channels/1')
            if webcam_url:
                infer_camera(self.model, webcam_url=webcam_url, conf=self.confidence)

    def _setup_sidebar(self):
        self.__model_config()
        self.__image_video_config()

    def _setup_main(self):
        st.title("Interface for Bộ Công An")


if __name__ == "__main__":
    gui = GUI()
