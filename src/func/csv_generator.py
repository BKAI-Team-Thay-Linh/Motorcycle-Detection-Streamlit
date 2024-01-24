import os
import pandas as pd
def txt_to_csv():
    """
    Tạo file CSV từ các file TXT trong thư mục chỉ định.
    CSV columns: 'frame_name', 'class', 'x', 'y', 'width', 'height'
    """
    # Dữ liệu cho DataFrame
    data = []
    directory = './runs/detect/predict/labels/'


    # Duyệt qua mỗi file TXT trong thư mục
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            # Đổi tên frame từ .txt sang .jpg
            frame_name = file.replace(".txt", ".jpg")

            # Đọc file TXT
            with open(os.path.join(directory, file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # Thêm dữ liệu vào danh sách
                        class_id, x, y, width, height = parts
                        data.append([frame_name, class_id, x, y, width, height])

    # Tạo DataFrame
    df = pd.DataFrame(data, columns=['frame_name', 'class', 'x', 'y', 'width', 'height'])
    # Lưu DataFrame thành file CSV
    csv_path = 'annotation/annotations.csv'
    df.to_csv(csv_path, index=False)

    return csv_path



def update_csv_classes():
    """
    Cập nhật trường 'class' trong file CSV dựa trên các file TXT.

    Parameters:
    csv_path (str): Đường dẫn đến file CSV.
    txt_folder (str): Đường dẫn đến thư mục chứa các file TXT.
    """
    # Đọc file CSV
    csv_path = 'annotation/annotations_sorted.csv'
    txt_folder = 'runs/classify/predict/labels'
    df = pd.read_csv(csv_path)
    # Cập nhật trường 'class' cho mỗi dòng trong DataFrame
    for i, file in enumerate(sorted(os.listdir(txt_folder))):
        if file.endswith(".txt"):
            # Đọc file TXT
            with open(os.path.join(txt_folder, file), 'r') as f:
                line = f.readline()
                new_class = line.strip().split()[1]

                # Cập nhật class mới vào DataFrame
                if i < len(df):
                    df.at[i, 'class'] = new_class

    # Lưu DataFrame đã được cập nhật vào file mới
    updated_csv_path = csv_path.replace('.csv', '_updated.csv')
    df.to_csv(updated_csv_path, index=False)

    return updated_csv_path


