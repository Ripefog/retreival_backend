from pymilvus import connections, Collection
import json

# 1. Kết nối đến Milvus
connections.connect(
    alias="default",
    host="localhost",  # đổi thành host của bạn
    port="19530"
)

# 2. Load collection
collection_name = "your_collection_name"  # đổi thành tên collection của bạn
collection = Collection(collection_name)

# 3. Truy vấn theo keyframe_id
keyframe_id = "L01_V011"
results = collection.query(
    expr=f'keyframe_id == "{keyframe_id}"',
    output_fields=["label"],  # chỉ lấy label, hoặc thêm "object", "color" nếu có
    consistency_level="Strong"
)

# 4. Xử lý kết quả
if not results:
    print(f"Không tìm thấy dữ liệu cho {keyframe_id}")
else:
    for r in results:
        label_data = r.get("label", None)
        if label_data:
            try:
                # Nếu label là chuỗi JSON chứa object và color
                label_json = json.loads(label_data)
                obj = label_json.get("object")
                color = label_json.get("color")
                print(f"Object: {obj}, Color: {color}")
            except json.JSONDecodeError:
                print(f"Nội dung label không phải JSON: {label_data}")
        else:
            print("Không có trường label.")