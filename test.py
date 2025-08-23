import os

folder_path = r"D:\HCM_AI_CHALLENGE\KNBHT\public\assets\keyframes\keyframes_output"  # Đường dẫn tới folder ảnh

for filename in os.listdir(folder_path):
    if filename.startswith("L02_"):
        new_filename = filename.replace("L02_", "", 1)  # chỉ thay thế lần đầu tiên
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Đổi: {filename} -> {new_filename}")
