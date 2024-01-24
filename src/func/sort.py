import os
import pandas as pd

def sort_files_numerically():
    """
    Sắp xếp các file trong thư mục theo thứ tự số tăng dần dựa trên tên file.

    Parameters:
    directory (str): Đường dẫn đến thư mục chứa các file.

    Returns:
    list: Danh sách các tên file đã được sắp xếp.
    """
    directory = 'runs/detect/predict/labels'

    # Lấy danh sách các file trong thư mục
    files = os.listdir(directory)

    # Sắp xếp các file dựa trên phần số trong tên file
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    return sorted_files

# Sử dụng hàm


def sort_csv_by_filename():
    """
    Sắp xếp file CSV tăng dần theo trường đầu tiên chứa tên file.

    Parameters:
    csv_path (str): Đường dẫn đến file CSV.

    Returns:
    str: Đường dẫn đến file CSV đã được sắp xếp.
    """
    # Đọc file CSV
    csv_path ='annotation/annotations.csv'

    df = pd.read_csv(csv_path)

    # Sắp xếp DataFrame dựa trên trường đầu tiên, sau khi chuyển đổi nó thành số
    df['sort_key'] = df.iloc[:, 0].apply(lambda x: int(x.split('_')[1].split('.')[0]))
    df_sorted = df.sort_values('sort_key').drop('sort_key', axis=1)

    # Lưu DataFrame đã được sắp xếp vào file mới
    sorted_csv_path = csv_path.replace('.csv', '_sorted.csv')
    df_sorted.to_csv(sorted_csv_path, index=False)

    return sorted_csv_path

