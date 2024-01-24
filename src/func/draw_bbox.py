import cv2
import os
from pandas import pd

def draw_bounding_box(image, class_id, x_center, y_center, width, height):
    """
    Vẽ bounding box trên ảnh với thông tin class tương ứng và cấu hình vẽ tùy chỉnh.

    Parameters:
    image (numpy.ndarray): Ảnh để vẽ bounding box.
    class_id (int): ID của class (0 hoặc 1).
    x_center, y_center (float): Tọa độ tâm của bounding box (tỉ lệ, không phải pixel).
    width, height (float): Chiều rộng và chiều cao của bounding box (tỉ lệ, không phải pixel).
    thickness (int): Độ dày của đường viền bounding box.
    font_scale (float): Kích thước của phông chữ cho title label.
    
    Returns:
    numpy.ndarray: Ảnh đã vẽ bounding box.
    """

    thickness=3
    font_scale=1
    # Convert fractional coordinates to pixel coordinates
    img_height, img_width, _ = image.shape
    box_width = int(width * img_width)
    box_height = int(height * img_height)
    x_min_px = int(round((x_center * img_width) - (box_width / 2)))
    y_min_px = int(round((y_center * img_height) - (box_height / 2)))
    x_max_px = x_min_px + box_width
    y_max_px = y_min_px + box_height

    # Define the color based on the class_id
    color = (255, 0, 0) if class_id == 0 else (0, 255, 0)
    label = "xe số" if class_id == 0 else "xe ga"

    # Draw the bounding box on the image
    cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), color, thickness)

    # Draw the label with a background rectangle
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y_min_label = max(y_min_px, label_size[1])  # Ensure the label is within the top boundary
    cv2.rectangle(image, (x_min_px, y_min_label - label_size[1]), (x_min_px + label_size[0], y_min_label + base_line), color, cv2.FILLED)
    cv2.putText(image, label, (x_min_px, y_min_label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

    return image

def draw_bounding_boxes_from_csv():
    """
    Vẽ bounding boxes lên các frame dựa trên thông tin từ file CSV và lưu vào thư mục mới.

    Parameters:
    input_folder_path (str): Đường dẫn đến thư mục chứa các frame gốc.
    output_folder_path (str): Đường dẫn đến thư mục nơi lưu các frame đã vẽ bounding box.
    csv_file (str): Đường dẫn đến file CSV chứa thông tin annotations.
    """
    input_folder_path = 'src/frame'
    output_folder_path = 'src/solved_frame'
    csv_file = 'annotation/annotations_sorted_updated.csv'

    # Tạo thư mục output nếu nó không tồn tại
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

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
            continue
        
        # Lấy tất cả các annotations cho frame hiện tại
        annotations = df[df['frame_name'] == frame]

        # Vẽ mỗi bounding box lên frame
        for _, row in annotations.iterrows():
            class_id, x, y, width, height = row['class'], row['x'], row['y'], row['width'], row['height']
            image = draw_bounding_box(image, class_id, x, y, width, height)
        
        # Lưu ảnh đã được annotate vào thư mục output
        cv2.imwrite(output_frame_path, image)


# input_folder_path = 'Test folder'
# output_folder_path = 'Last_demo'
# draw_bounding_boxes_from_csv(input_folder_path, output_folder_path, csv_file_path)
