import os
import cv2
import glob
from tqdm import tqdm
import traceback
import torch

# Import các module đã tạo
from config import *
# from process import *
# from schema import *
# from .database import init_database, close_database
from retrieval_engine import *
# Patch cho torch.meshgrid nếu cần thiết để tương thích với các phiên bản cũ hơn
if not hasattr(torch, 'meshgrid') or 'indexing' not in torch.meshgrid.__code__.co_varnames:
    original_meshgrid = torch.meshgrid


    def patched_meshgrid(*tensors, **kwargs):
        kwargs.pop('indexing', None)
        return original_meshgrid(*tensors, **kwargs)


    torch.meshgrid = patched_meshgrid


def parse_keyframe_info(frame_path: str):
    """
    Phân tích video_id và timestamp từ đường dẫn keyframe.
    Ví dụ: "L01_V001_0012.34s.jpg" -> video_id="L01_V001", timestamp=12.34
    """
    try:
        filename = os.path.basename(frame_path)
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')

        timestamp_str = parts[-1]
        timestamp = float(timestamp_str.replace('s', ''))

        # Tái tạo video_id từ các phần còn lại
        video_id = '_'.join(parts[:-1])
        return video_id, timestamp
    except Exception as e:
        print(f"Warning: Could not parse info from {frame_path}. Using fallback. Error: {e}")
        # Fallback an toàn hơn
        parts = frame_path.replace("\\", "/").split('/')
        video_id = parts[-2] if len(parts) > 1 else "unknown_video"
        return video_id, 0.0


if __name__ == '__main__':
    print("\n--- [START] MILVUS KEYFRAME VECTOR PROCESSING PIPELINE ---")

    # 1. KHỞI TẠO CÁC THÀNH PHẦN
    config = Settings()
    os.makedirs(config.KEYFRAME_ROOT_DIR, exist_ok=True)

    print("Initializing Milvus managers...")
    managers = {
        'clip': MilvusManager(config.CLIP_COLLECTION, config),
        'beit3': MilvusManager(config.BEIT3_COLLECTION, config),
        'object': MilvusManager(config.OBJECT_COLLECTION, config),
        'color': MilvusManager(config.COLOR_COLLECTION, config),
    }

    print("Initializing feature extractors...")
    feature_extractor = FeatureExtractor(config.DEVICE)
    object_color_detector = ObjectColorDetector(config.DEVICE)

    # keyframe_files = glob.glob(os.path.join(config.KEYFRAME_ROOT_DIR, "*.jpg"))

    keyframe_files = []
    for i in range(config.MILVUS_START, config.MILVUS_END + 1):
        folder_name = f"V{i:03d}"  # Tạo tên dạng V001, V002...
        folder_path = os.path.join(config.KEYFRAME_ROOT_DIR, folder_name)
        if os.path.isdir(folder_path):
            keyframe_files.extend(
                glob.glob(os.path.join(folder_path, "*.jpg"))
            )

    if not keyframe_files:
        print(f"!!! ERROR: No keyframes found in '{config.KEYFRAME_ROOT_DIR}'. Pipeline cannot continue.")
        exit(1)
    else:
        print(f"Found {len(keyframe_files)} keyframes to process.")

    # 2. BẮT ĐẦU VÒNG LẶP XỬ LÝ
    processed_count = 0
    error_count = 0

    for frame_path in tqdm(keyframe_files, desc="Processing Keyframes for Milvus"):
        try:
            # === BƯỚC A: PHÂN TÍCH THÔNG TIN ===
            video_id, timestamp = parse_keyframe_info(frame_path)
            # Dùng tên file đầy đủ làm ID duy nhất, mạnh mẽ và tường minh
            keyframe_id = os.path.basename(frame_path)

            # === BƯỚC B: ĐỌC VÀ TRÍCH XUẤT ĐẶC TRƯNG ẢNH ===
            frame_cv = cv2.imread(frame_path)
            if frame_cv is None:
                print(f"Error: Cannot read image: {frame_path}")
                error_count += 1
                continue

            clip_img_emb, beit3_img_emb = feature_extractor.get_image_embeddings(frame_cv)

            # === BƯỚC C: INSERT VECTOR ẢNH VÀO MILVUS ===
            managers['clip'].insert([
                [keyframe_id],
                [clip_img_emb.tolist()],
                [video_id],
                [timestamp]
            ])
            # print(beit3_img_emb.tolist())
            managers['beit3'].insert([
                [keyframe_id],
                [beit3_img_emb.tolist()],
                [video_id],
                [timestamp]
            ])

            # === BƯỚC D: PHÁT HIỆN ĐỐI TƯỢNG VÀ MÀU SẮC ===
            dominant_colors_lab, object_colors_lab = object_color_detector.detect(frame_path)

            # === BƯỚC E: INSERT VECTOR CHO OBJECT/COLOR VÀO MILVUS ===
            # Sử dụng CLIP để mã hóa text vì nó mạnh hơn cho các khái niệm đơn lẻ
            for label in object_colors_lab.keys():
                # Chuyển tên đối tượng thành embedding (CLIP)
                obj_emb = feature_extractor.get_clip_text_embedding(label)

                # Lặp qua từng màu sắc của đối tượng (dưới dạng LAB)
                for value in object_colors_lab[label]:
                    color, bbox = value
                    valuee = ",".join(map(str, list(color) + list(bbox)))
                    object_entry_id = f"{keyframe_id}_obj_{label}"
                    managers['object'].insert([
                        [object_entry_id],
                        [obj_emb.tolist()],  # Đưa embedding của đối tượng
                        [video_id],
                        [timestamp],
                        [valuee]  # Lưu màu LAB dưới dạng list (L, A, B)
                    ])

            # Lưu màu sắc chủ đạo (dominant colors) vào Milvus
            for i, color in enumerate(dominant_colors_lab):
                # Tạo PK duy nhất cho mỗi entry color (tính theo index)
                color_entry_id = f"{keyframe_id}_color_{i}"
                managers['color'].insert([
                    [color_entry_id],
                    [list(color)],  # Lưu màu LAB dưới dạng list (L, A, B)
                    [video_id],
                    [timestamp],
                    ["alo"]
                ])

            processed_count += 1
            if processed_count % 100 == 0:
                print(f" -> Processed {processed_count}/{len(keyframe_files)} keyframes...")

        except Exception as e:
            print(f"\n--- FATAL ERROR while processing {frame_path} ---")
            traceback.print_exc()
            error_count += 1
            continue

    # 3. KẾT THÚC VÀ THỐNG KÊ
    print(f"\n--- [FINISHED] MILVUS KEYFRAME VECTOR PROCESSING PIPELINE ---")
    print(f"Successfully processed: {processed_count} keyframes")
    print(f"Errors encountered: {error_count} keyframes")

    print("Flushing all Milvus collections to ensure data is written...")
    for name, manager in managers.items():
        try:
            manager.collection.flush()
            print(f"[✔] Flushed '{name}' collection")
        except Exception as e:
            print(f"[❌] Error flushing '{name}' collection: {e}")

    print("\nPipeline completed!")