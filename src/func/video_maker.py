import cv2
import os

def create_video_from_frames(input_folder_path, output_folder_path, output_video_name, fps=30):
    """
    Tạo video từ các frame liên tục trong một thư mục gốc và lưu vào thư mục đích.

    Parameters:
    input_folder_path (str): Đường dẫn đến thư mục chứa các frame gốc.
    output_folder_path (str): Đường dẫn đến thư mục nơi lưu video.
    output_video_name (str): Tên file video đầu ra.
    fps (int): Số khung hình trên giây của video.
    """
    # Tạo thư mục output nếu nó không tồn tại
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Lấy danh sách tên file các frame và sắp xếp
    frame_files = sorted([f for f in os.listdir(input_folder_path) if f.endswith('.jpg')],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Đọc một frame đầu tiên để xác định thông số cho VideoWriter
    first_frame = cv2.imread(os.path.join(input_folder_path, frame_files[0]))
    height, width, layers = first_frame.shape

    # Khởi tạo VideoWriter
    output_video_path = os.path.join(output_folder_path, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng codec
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Duyệt qua mỗi frame và thêm vào video
    for frame_file in frame_files:
        frame_path = os.path.join(input_folder_path, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Giải phóng VideoWriter khi hoàn thành
    video_writer.release()
    print(f"Video đã được tạo và lưu tại: {output_video_path}")

    return output_video_path

# Ví dụ sử dụng hàm:
input_folder_path = 'Last_demo'
output_folder_path = 'An'
output_video_name = 'output_video.mp4'
create_video_from_frames(input_folder_path, output_folder_path, output_video_name)
