import os
import shutil

def remove_folder_contents(folder):
    """
    Remove all files and subdirectories in specified folder if it exists.
    Parameters:
        folder (str): The path to the folder to clear.
    """
    if not os.path.exists(folder):
        print(f"The folder {folder} does not exist. Skipping removal.")
        return True

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
            return False
    return True


# Example usage

