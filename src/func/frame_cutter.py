import cv2
import os

def extract_and_save_frames(video_stream_path, save_folder):
    """
    Tách stream thành frames với tần suất 30 frames/giây và lưu chúng vào một thư mục.

    Parameters:
    video_stream_path (str): Đường dẫn tới video stream hoặc thiết bị camera.
    save_folder (str): Đường dẫn tới thư mục nơi lưu các frames.
    """
    # Mở video stream
    cap = cv2.VideoCapture(video_stream_path)

    # Kiểm tra xem stream có mở được không
    if not cap.isOpened():
        print("Không thể mở video stream hoặc file.")
        return

    # Tạo thư mục nếu nó không tồn tại
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    frame_count = 0
    interval = int(1000 / 90)  # Khoảng thời gian giữa mỗi frame (ms)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Dừng nếu không đọc được dữ liệu nữa

        # Lưu frame
        frame_file = os.path.join(save_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_file, frame)
        frame_count += 1

        # Hiển thị frame
# Khi xong, giải phóng và đóng tất cả cửa sổ

# Sử dụng hàm
# video_stream_path = 'rtsp://Cam2:Etcop2@2023Ai2@Cam26hc.cameraddns.net:556/Streaming/Channels/1'
# save_folder_path = 'Test folder'
# extract_and_save_frames(video_stream_path, save_folder_path)
