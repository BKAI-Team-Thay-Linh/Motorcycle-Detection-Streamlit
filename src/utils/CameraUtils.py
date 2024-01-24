import os
import cv2


def process_webcam(webcam_url: str, save_folder: str, fps: int = 30, video_length: int = 10):
    """
    Extract frames from a video source and save it to the specified folder
    Args:
        `webcam_url (str)`: The url of the webcam source
        `save_folder (str)`: The folder to save the extracted frames
        `fps (int, optional)`: The number of frames will be saved in a second. Defaults to 30.
        `video_length (int, optional)`: The video length of a batch. Defaults to 10. The stream 
        will be extracted frames for 10 seconds, then process them to make a video with bounding boxes
        and classes on them, then play the video for 10 seconds. While the video is playing, the stream will
        extract frames for the next batch.
    """
    pass
