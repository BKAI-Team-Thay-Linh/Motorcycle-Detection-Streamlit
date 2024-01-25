from queue import Queue
import time
from typing import List
from PIL import Image
from ultralytics import YOLO
import cv2
import threading
import multiprocessing as mp


class Demo():
    def __init__(self, stream_url: str, fps: int = 30, conf: float = 0.4):
        self.extracted_frame = Queue()
        self.detected_frame = Queue()
        self.small_images = Queue()
        self.predicted_frame = Queue()
        self.done_frame = Queue()

        self.map_frame_with_bb_coor: dict[List[tuple]] = {}
        self.map_frame_with_bb_image: dict = {}
        self.map_bb_image_with_frame: dict = {}
        self.bb_image_info: dict = {}

        self.is_playing = False
        self.stream_url = stream_url
        self.fps = fps
        self.conf = conf

        # Model to detect bounding boxes
        self.detection_model = YOLO(model='src/configs/weights/detection/yolov8m.pt').to('cuda')

    def extract_frame_thread(self, cap: cv2.VideoCapture):
        interval = int(1000 / self.fps)

        while self.is_playing:
            ret, frame = cap.read()
            if not ret:
                print("\nUnable to read frame. Stopping...")
                cap.release()
                break

            # Save frame
            self.extracted_frame.put(frame)

            # # Show frame
            # cv2.imshow("Webcam", frame)

            time.sleep(interval / 1000)

    def detect_frame_thread(self):
        while self.is_playing:
            if self.extracted_frame.empty():
                continue

            frame = self.extracted_frame.get()
            print(f"==>> frame: {frame}")
            detection_result = self.detection_model.predict(frame, conf=self.conf, classes=3)
            self.done_frame.put(detection_result[0].plot())  # DEMO the detection result

    def output_frame_thread(self):
        while self.is_playing:
            if self.done_frame.empty():
                continue

            frame = self.done_frame.get()
            cv2.imshow("Webcam", frame)

            time.sleep(1 / self.fps)

    def run(self):
        self.is_playing = True

        # Capture frames from the camera
        cap = cv2.VideoCapture(self.stream_url)

        # Thread to extract frames
        extract_frame_thread = threading.Thread(target=self.extract_frame_thread, args=(cap,))
        extract_frame_thread.start()

        # Thread to detect frames
        detect_frame_thread = threading.Thread(target=self.detect_frame_thread)
        detect_frame_thread.start()

        # Thread to output frames
        output_frame_thread = threading.Thread(target=self.output_frame_thread)
        output_frame_thread.start()


if __name__ == '__main__':
    # demo = Demo(
    #     stream_url='rtsp://Cam2:Etcop2@2023Ai2@Cam26hc.cameraddns.net:556/Streaming/Channels/1',
    #     fps=15
    # )

    demo = Demo(
        stream_url='sampel.mp4',
        fps=15
    )

    demo.run()
