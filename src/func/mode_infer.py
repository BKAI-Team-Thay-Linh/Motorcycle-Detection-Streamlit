from ultralytics import YOLO
import streamlit as st
import cv2
import fnmatch
import time
import os
def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image,  (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf, classes=3, save_crop = True, save_txt = True, save = True)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    
def _display_classify_frame(model, st_frame, folder):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    for file in os.listdir(folder): 
        if fnmatch.fnmatch(file, '*.jpg'):  # Kiểm tra nếu file là một file .jpg
                            # Loop through each file and display the classified frame
            path = os.path.join(folder, file)
                                
                        # Check if the image was read successfully
            if file is not None:
                results = model.predict(path, save_txt = True)
                time.sleep(0.5)

def detect_on_stream(conf, model, folder):
    for file in os.listdir(folder): 
        if fnmatch.fnmatch(file, '*.jpg'):  # Kiểm tra nếu file là một file .jpg
                        # Loop through each file and display the classified frame
            path = os.path.join(folder, file)
            model.predict(path, conf = 0.4, save_crop = True, save_txt = True, classes = 3)

def _yolo_classify():
    model_path = './weights/classify/best.pt'
    test_folder= 'runs/detect/predict/crops/motorcycle'
    model = YOLO(model_path)

    for file in os.listdir(test_folder): 
        if fnmatch.fnmatch(file, '*.jpg'):  # Kiểm tra nếu file là một file .jpg
                            # Loop through each file and display the classified frame
            path = os.path.join(test_folder, file)
            model.predict(path, conf = 0.6, save_txt = True)