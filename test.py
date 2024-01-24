import os
import shutil
import time
import cv2

from ultralytics import YOLO


def extract_and_save_frames(webcam_url: str, save_folder: str, fps: int = 30, video_length: int = 10):
    cap = cv2.VideoCapture(webcam_url)

    if not cap.isOpened():
        raise Exception("Unable to open webcam")

    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)

    frame_count = 0
    time_elapsed = video_length * 1000  # in milliseconds
    interval = int(1000 / fps)

    print(f'Extracting Frames in {video_length} seconds ...')

    while time_elapsed > 0:
        ret, frame = cap.read()
        if not ret:
            print("\nUnable to read frame. Stopping...")
            cap.release()
            break

        # Save frame
        frame_file = os.path.join(save_folder, f"{frame_count}.jpg")
        cv2.imwrite(frame_file, frame)
        frame_count += 1

        # Show frame
        cv2.imshow("Webcam", frame)

        # Update time elapsed
        print(f"Time elapsed: {time_elapsed:>5}ms", end='\r')
        time_elapsed -= interval

        if cv2.waitKey(interval) & 0xFF == ord('q'):
            print("\nStopping...")
            cap.release()
            break

    print("\nDone")
    cap.release()


def detect_bb(model_path: str, frames_folder: str, conf: float = 0.5):
    frames = os.listdir(frames_folder)
    model = YOLO(model_path)

    print(f'Starting detection using model: {os.path.basename(model_path)}...')

    for frame in frames:
        print(f'Processing {frame} ...')
        if frame.endswith('.jpg'):
            frame_path = os.path.join(frames_folder, frame)
            model.predict(frame_path, conf=conf, save_txt=True, save_crop=True, classes=3)


if __name__ == "__main__":
    video_stream_path = 'rtsp://Cam2:Etcop2@2023Ai2@Cam26hc.cameraddns.net:556/Streaming/Channels/1'

    extract_and_save_frames(
        webcam_url=video_stream_path,
        save_folder='.temp/webcam_frame',
        fps=30,
        video_length=10
    )

    detect_bb(
        model_path='src/configs/weights/detection/yolov8m.pt',
        frames_folder='.temp/webcam_frame',
        conf=0.5
    )
