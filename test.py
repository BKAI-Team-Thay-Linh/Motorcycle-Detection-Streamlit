import os
import shutil
import threading
import time
import cv2
import pandas as pd

from ultralytics import YOLO
from ultralytics.utils.files import WorkingDirectory
from PIL import Image

# Change working directory to the root of the project
WorkingDirectory(os.getcwd())


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
        frame_file = os.path.join(save_folder, f"frame_{frame_count}.jpg")
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
    start_time = time.time()

    shutil.rmtree('runs/detect', ignore_errors=True)

    for frame in frames:
        if frame.endswith('.jpg'):
            frame_path = os.path.join(frames_folder, frame)
            model.predict(frame_path, conf=conf, save_txt=True, save_crop=True, classes=3)

    print(f'Done. Total time: {time.time() - start_time:.2f}s')


def classify_bb(model_path: str, crops_folder: str, conf: float = 0.6):
    frames = os.listdir(crops_folder)
    model = YOLO(model_path)

    print(f'Starting classification using model: {os.path.basename(model_path)}...')
    start_time = time.time()

    shutil.rmtree('runs/classify', ignore_errors=True)

    for frame in frames:
        if frame.endswith('.jpg'):
            frame_path = os.path.join(crops_folder, frame)
            model.predict(frame_path, conf=conf, save_txt=True)

    print(f'Done. Total time: {time.time() - start_time:.2f}s')


def labels_to_csv(detect_labels_folder: str, classify_labels_folder: str,  csv_path: str):
    data = []

    print(f'Converting labels in {detect_labels_folder} to csv ...')

    for label_file in os.listdir(detect_labels_folder):
        if label_file.endswith('.txt'):
            frame_name = label_file.replace('.txt', '.jpg')
            label_path = os.path.join(detect_labels_folder, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = parts
                        data.append([frame_name, class_id, x, y, w, h])

    # Create df
    df = pd.DataFrame(data, columns=['frame_name', 'class', 'x', 'y', 'width', 'height'])

    df['sort_key'] = df.iloc[:, 0].apply(lambda x: int(x.split('_')[1].split('.')[0]))
    df_sorted = df.sort_values('sort_key').drop('sort_key', axis=1)

    for i, file in enumerate(sorted(os.listdir(classify_labels_folder))):
        if file.endswith('.txt'):
            label_path = os.path.join(classify_labels_folder, file)
            with open(label_path, 'r') as f:
                line = f.readline()
                new_class = line.strip().split()[1]
                if i < len(df):
                    df_sorted.at[i, 'class'] = new_class

    df_sorted.to_csv(csv_path, index=False)

    print(f'Done. Saved to {csv_path}')


def _draw_bb(image, class_id, x, y, width, height):
    thickness = 2
    font_scale = 1

    # Convert fractional coordinates to pixel coordinates
    img_height, img_width, _ = image.shape
    box_width = int(width * img_width)
    box_height = int(height * img_height)
    x_min_px = int(round((x * img_width) - (box_width / 2)))
    y_min_px = int(round((y * img_height) - (box_height / 2)))
    x_max_px = x_min_px + box_width
    y_max_px = y_min_px + box_height

    # Define the color based on the class_id
    color = (255, 0, 0) if class_id == 0 else (0, 255, 0)
    label = "xe so" if class_id == 0 else "xe ga"

    # Draw the bounding box on the image
    cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), color, thickness)

    # Draw the label with a background rectangle
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y_min_label = max(y_min_px, label_size[1])  # Ensure the label is within the top boundary
    cv2.rectangle(image, (x_min_px, y_min_label - label_size[1]),
                  (x_min_px + label_size[0], y_min_label + base_line), color, cv2.FILLED)
    cv2.putText(image, label, (x_min_px, y_min_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return image


def draw_bounding_boxes_from_csv(input_folder_path, output_folder_path, csv_file):
    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path, exist_ok=True)

    # Đọc dữ liệu từ CSV
    df = pd.read_csv(csv_file)

    # Lọc ra danh sách các frame duy nhất
    unique_frames = df['frame_name'].unique()

    # Duyệt qua mỗi frame
    for frame in unique_frames:
        input_frame_path = os.path.join(input_folder_path, frame)
        output_frame_path = os.path.join(output_folder_path, frame)

        # Đọc ảnh frame
        image = cv2.imread(input_frame_path)
        if image is None:
            print(f" Cannot read image from path: {input_frame_path}")
            continue

        # Lấy tất cả các annotations cho frame hiện tại
        annotations = df[df['frame_name'] == frame]

        # Vẽ mỗi bounding box lên frame
        for _, row in annotations.iterrows():
            class_id, x, y, width, height = row['class'], row['x'], row['y'], row['width'], row['height']
            image = _draw_bb(image, class_id, x, y, width, height)

        # Lưu ảnh đã được annotate vào thư mục output
        cv2.imwrite(output_frame_path, image)
        print(f"Frame is saved at: {output_frame_path}", end='\r')


def create_video_from_frames(input_folder_path, output_folder_path, fps: int = 30):
    save_dir = os.path.dirname(output_folder_path)
    os.makedirs(save_dir, exist_ok=True)

    # Get all frame files
    frame_files = sorted([f for f in os.listdir(input_folder_path) if f.endswith('.jpg')],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))

    first_frame = cv2.imread(os.path.join(input_folder_path, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_folder_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(input_folder_path, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)

    video.release()
    print(f"Video is saved at: {output_folder_path}")

    return output_folder_path


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
        conf=0.4
    )

    classify_bb(
        model_path='src/configs/weights/classify/best.pt',
        crops_folder='runs/detect/predict/crops/motorcycle',
        conf=0.6
    )

    labels_to_csv(
        detect_labels_folder='runs/detect/predict/labels',
        classify_labels_folder='runs/classify/predict/labels',
        csv_path='annotations/predict.csv'
    )

    draw_bounding_boxes_from_csv(
        input_folder_path='.temp/webcam_frame',
        output_folder_path='.temp/webcam_frame_annotated',
        csv_file='annotations/predict.csv'
    )

    create_video_from_frames(
        input_folder_path='.temp/webcam_frame_annotated',
        output_folder_path='.temp/output/video.mp4',
        fps=15
    )
