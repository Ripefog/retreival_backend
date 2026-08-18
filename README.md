# Video Retrieval Backend

Backend API cho hệ thống tìm kiếm video đa phương thức với hybrid retrieval engine kết hợp CLIP và BEIT-3 models.

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Milvus        │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Elasticsearch  │
                       │  (Text Search)  │
                       └─────────────────┘
```

## 🚀 Tính năng chính

### 1. Hybrid Retrieval Engine

- **CLIP Model**: Semantic search với text-image matching
- **BEIT-3 Model**: Advanced vision-language model
- **Hybrid Approach**: Kết hợp cả hai với "filter and refine" strategy

### 2. Multi-modal Search

- **Text Search**: Natural language queries
- **Object Detection**: Filter theo objects trong video
- **Color Filtering**: Filter theo màu sắc
- **OCR Search**: Tìm kiếm text trong images
- **ASR Search**: Tìm kiếm speech transcripts

### 3. Database Integration

- **Milvus**: Vector database cho embeddings
- **Elasticsearch**: Text search cho OCR/ASR
- **Ngrok Tunnels**: Remote access support

### 4. UI Input Processing

- **Text Processing**: Xử lý và chuẩn hóa text input
- **Query Suggestions**: Gợi ý query tự động
- **Filter Validation**: Validate và xử lý filters
- **Batch Processing**: Xử lý nhiều input cùng lúc

## 📦 Cài đặt

### Yêu cầu hệ thống

- Python 3.9+
- Docker (tùy chọn)
- CUDA (tùy chọn, cho GPU acceleration)

### 1. Clone repository

```bash
git clone <repository-url>
cd video-retrieval-backend
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình environment

Tạo file `.env`:

```env
# Milvus Configuration (Cloud)
MILVUS_URI=https://in03-e40f88db343fc76.serverless.aws-eu-central-1.cloud.zilliz.com
MILVUS_TOKEN=b7c8cee41ac36a48967f63a899e22b4ec6d6f2a33cedb8b0b72bbee43fc28bfcf1a109b171703871837b08d45e0fc14697a6a770
MILVUS_ALIAS=default

# Elasticsearch Configuration (via ngrok)
ELASTICSEARCH_HOST=0.tcp.ap.ngrok.io
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_USE_SSL=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Model Configuration
DEVICE=cuda  # hoặc cpu
```

## 🚀 Chạy ứng dụng

### 🟢 Bước 1: Khởi chạy Backend Server

Mở terminal và chạy các lệnh sau:

```bash
cd /home/s12gbn3/workspace/retreival_backend
source /home/s12gbn3/pytorchenv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend sẽ tự động nạp cấu hình Zilliz Cloud Milvus từ file `.env` và lắng nghe tại: `http://localhost:8000`.

---

### 🌐 Bước 2: (Tùy chọn) Mở ra Internet cho Frontend kết nối

Nếu Frontend nằm ở máy khác / Cloud / Vercel, bạn mở thêm 1 tab terminal khác và chạy:

```bash
cloudflared tunnel --url http://localhost:8000
```

Terminal sẽ tạo ra 1 đường dẫn HTTPS công khai dạng: `https://xxx-xxx-xxx.trycloudflare.com` để Frontend kết nối vào.

---

### 🛠️ Các lệnh thay thế khác

#### Production mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Docker (tùy chọn)

```bash
# Build base image (chỉ 1 lần)
docker build -t video-retrieval-backend-base --target base .

# Build runtime image
docker build -t video-retrieval-backend --target runtime .

docker run --gpus all -p 8000:8000 video-retrieval-backend
```

## 📡 API Endpoints

### Health Check

```bash
GET /health
```

Response:

```json
{
  "status": "healthy",
  "milvus": {
    "status": "connected",
    "connected": true,
    "collections": {
      "arch_clip_image_v3": {
        "num_entities": 1000000,
        "loaded": true
      }
    }
  },
  "elasticsearch": {
    "status": "connected",
    "connected": true,
    "indices": {
      "video_retrieval_metadata_v3": {
        "exists": true,
        "doc_count": 500000
      }
    }
  },
  "retriever": "initialized"
}
```

### Text Search

```bash
POST /search
```

Request:

```json
{
  "text_query": "person walking in red shirt",
  "mode": "hybrid",
  "object_filters": ["person", "shirt"],
  "color_filters": ["red"],
  "ocr_query": "text in image",
  "asr_query": "speech content",
  "top_k": 100
}
```

Response:

```json
{
  "query": "person walking in red shirt",
  "mode": "hybrid",
  "results": [
    {
      "keyframe_id": "frame_000123",
      "video_id": "video_001",
      "timestamp": 45.2,
      "score": 0.95,
      "reasons": [
        "Matched query: person walking in red shirt",
        "Object filter: person",
        "Color filter: red",
        "Hybrid reranking applied"
      ],
      "metadata": {
        "rank": 1,
        "collection": "arch_clip_image_v3",
        "confidence": 0.95
      }
    }
  ],
  "total_results": 1,
  "search_time": 0.15
}
```

### Search Modes

```bash
GET /search/modes
```

Response:

```json
{
  "modes": ["hybrid", "clip", "beit3"],
  "descriptions": {
    "hybrid": "Kết hợp CLIP và BEIT-3 cho kết quả tốt nhất",
    "clip": "Chỉ sử dụng CLIP model",
    "beit3": "Chỉ sử dụng BEIT-3 model"
  }
}
```

### Collections Info

```bash
GET /collections
```

Response:

```json
{
  "collections": {
    "arch_clip_image_v3": {
      "num_entities": 1000000,
      "schema": "CollectionSchema(...)",
      "loaded": true
    },
    "arch_beit3_image_v3": {
      "num_entities": 1000000,
      "schema": "CollectionSchema(...)",
      "loaded": true
    }
  }
}
```

### Compare Search Modes

```bash
POST /search/compare
```

Request:

```json
{
  "text_query": "person walking",
  "top_k": 10
}
```

Response:

```json
{
  "query": "person walking",
  "comparison": {
    "hybrid": {
      "results": [...],
      "total_results": 10
    },
    "clip": {
      "results": [...],
      "total_results": 10
    },
    "beit3": {
      "results": [...],
      "total_results": 10
    }
  }
}
```

## 🖥️ UI Input APIs

### Process UI Input

```bash
POST /ui/input
```

Request:

```json
{
  "input_text": "person walking in red shirt",
  "input_type": "search",
  "user_id": "user123",
  "session_id": "session456"
}
```

Response:

```json
{
  "success": true,
  "message": "Input processed successfully",
  "processed_text": "person walking in red shirt",
  "suggestions": [
    "person walking in red shirt trong video",
    "tìm kiếm person walking in red shirt",
    "video có person walking in red shirt"
  ],
  "metadata": {
    "input_type": "search",
    "user_id": "user123",
    "session_id": "session456",
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

### Text Processing

```bash
POST /ui/text/process
```

Request:

```json
{
  "text": "Person walking in red shirt on sunny day",
  "processing_type": "extract",
  "language": "en"
}
```

Response:

```json
{
  "original_text": "Person walking in red shirt on sunny day",
  "processed_text": "person walking red shirt sunny day",
  "processing_type": "extract",
  "confidence": 0.95,
  "extracted_info": {
    "word_count": 7,
    "char_count": 35,
    "language": "en",
    "processing_type": "extract"
  }
}
```

### Query Suggestions

```bash
POST /ui/query/suggest
```

Request:

```json
{
  "partial_query": "person",
  "context": ["video", "walking"],
  "max_suggestions": 3
}
```

Response:

```json
{
  "partial_query": "person",
  "suggestions": ["person trong video", "tìm kiếm person", "video có person"],
  "confidence_scores": [0.9, 0.8, 0.7],
  "total_suggestions": 3
}
```

### Filter Input Processing

```bash
POST /ui/filter/input
```

Request:

```json
{
  "filter_type": "object",
  "filter_values": ["person", "car", "building"],
  "operator": "AND",
  "priority": 1
}
```

Response:

```json
{
  "filter_type": "object",
  "processed_filters": ["person", "car", "building"],
  "validation_status": true,
  "suggestions": ["person", "car", "animal", "building", "object"],
  "error_messages": null
}
```

### Batch Input Processing

```bash
POST /ui/batch/input
```

Request:

```json
{
  "inputs": ["person walking", "red car driving", "building with windows"],
  "batch_type": "search",
  "priority": "normal"
}
```

Response:

```json
{
  "total_inputs": 3,
  "processed_inputs": 3,
  "failed_inputs": 0,
  "results": [
    {
      "index": 0,
      "original_input": "person walking",
      "processed_input": "person walking",
      "status": "success",
      "message": "Processed successfully",
      "search_ready": true
    }
  ],
  "batch_id": "uuid-string",
  "processing_time": 0.123
}
```

### Get Input Types

```bash
GET /ui/input/types
```

Response:

```json
{
  "input_types": [
    {
      "type": "general",
      "description": "Input chung cho mọi mục đích",
      "supported_operations": ["process", "validate", "suggest"]
    },
    {
      "type": "search",
      "description": "Input cho tìm kiếm video",
      "supported_operations": ["process", "suggest", "expand"]
    }
  ],
  "processing_types": ["normalize", "extract", "analyze", "validate", "suggest"]
}
```

### Get Batch Status

```bash
GET /ui/input/status/{batch_id}
```

Response:

```json
{
  "batch_id": "uuid-string",
  "status": "completed",
  "progress": 100,
  "message": "Batch processing completed successfully"
}
```

## 🔧 Cấu hình

### Environment Variables

| Variable             | Description        | Default                                                                                                    |
| -------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `MILVUS_URI`         | Milvus Cloud URI   | `https://in03-e40f88db343fc76.serverless.aws-eu-central-1.cloud.zilliz.com`                                |
| `MILVUS_TOKEN`       | Milvus Cloud token | `b7c8cee41ac36a48967f63a899e22b4ec6d6f2a33cedb8b0b72bbee43fc28bfcf1a109b171703871837b08d45e0fc14697a6a770` |
| `ELASTICSEARCH_HOST` | Elasticsearch host | `0.tcp.ap.ngrok.io`                                                                                        |
| `ELASTICSEARCH_PORT` | Elasticsearch port | `9200`                                                                                                     |
| `DEVICE`             | Device for models  | `cuda`                                                                                                     |
| `API_HOST`           | API host           | `0.0.0.0`                                                                                                  |
| `API_PORT`           | API port           | `8000`                                                                                                     |

### Collections Configuration

| Collection            | Purpose           | Description                       |
| --------------------- | ----------------- | --------------------------------- |
| `arch_clip_image_v3`  | CLIP embeddings   | Vector embeddings từ CLIP model   |
| `arch_beit3_image_v3` | BEIT-3 embeddings | Vector embeddings từ BEIT-3 model |
| `arch_object_name_v3` | Object detection  | Object labels và bounding boxes   |
| `arch_color_name_v3`  | Color detection   | Color information                 |

### Indices Configuration

| Index                         | Purpose         | Description                   |
| ----------------------------- | --------------- | ----------------------------- |
| `video_retrieval_metadata_v3` | Video metadata  | Thông tin metadata của videos |
| `ocr`                         | OCR text        | Text extracted từ images      |
| `video_transcripts`           | ASR transcripts | Speech transcripts            |

## 🔍 Search Pipeline

### 1. Initial Candidate Retrieval

- Encode text query thành vector
- Search trong Milvus collections
- Lấy top-k candidates

### 2. Object/Color Filtering

- Filter candidates theo object labels
- Filter theo color information
- Áp dụng boolean logic

### 3. Text Filtering (OCR/ASR)

- Search trong Elasticsearch indices
- Filter theo OCR text matches
- Filter theo ASR transcript matches

### 4. Hybrid Reranking

- Kết hợp scores từ CLIP và BEIT-3
- Áp dụng weighted combination
- Sort theo final scores

### 5. Result Formatting

- Format kết quả cuối cùng
- Thêm metadata và reasons
- Return structured response

## 📊 Monitoring

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check Milvus connection
curl http://localhost:8000/collections
```

### Logs

```bash
# View application logs
tail -f logs/app.log

# View error logs
tail -f logs/error.log
```

## 🚀 Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
python app/main.py
```

### Production with Docker

```bash
# Build image
docker build -t video-retrieval-backend .

# Run container
docker run -d \
  -p 8000:8000 \
  -e MILVUS_HOST=your-milvus-host \
  -e ELASTICSEARCH_HOST=your-es-host \
  video-retrieval-backend
```

### Production with ngrok

```bash
# Install ngrok
pip install pyngrok

# Create tunnel
ngrok http 8000

# Update frontend URLs
export const API_URL = "https://your-ngrok-url";
```

## 🔧 Development

### Project Structure

```
video-retrieval-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models.py            # Pydantic models
│   ├── database.py          # Database connections
│   └── retrieval_engine.py  # Hybrid retriever
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .env                    # Environment variables
```

### Adding New Features

1. **New Search Mode**:

   - Add mode to `SearchMode` enum in `models.py`
   - Implement logic in `retrieval_engine.py`
   - Update API endpoints in `main.py`

2. **New Filter Type**:

   - Add filter field to `SearchRequest` in `models.py`
   - Implement filtering logic in `retrieval_engine.py`
   - Update search pipeline

3. **New Database**:

   - Add connection logic in `database.py`
   - Update configuration in `config.py`
   - Add health checks

4. **New UI Input Type**:
   - Add new model in `models.py`
   - Implement processing logic in `main.py`
   - Add corresponding test in `test_api.py`

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_retrieval.py::test_search

# Run UI input tests
python test_api.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Milvus Connection Failed**

   ```bash
   # Check Milvus Cloud connection
   curl http://localhost:8000/health

   # Verify token and URI
   echo $MILVUS_URI
   echo $MILVUS_TOKEN
   ```

2. **Elasticsearch Connection Failed**

   ```bash
   # Check ES tunnel
   curl -k https://your-es-ngrok-url/_cluster/health

   # Check indices
   curl -k https://your-es-ngrok-url/_cat/indices
   ```

3. **Model Loading Failed**

   ```bash
   # Check GPU availability
   nvidia-smi

   # Check CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Performance Tuning

1. **GPU Acceleration**

   ```bash
   # Enable CUDA
   export DEVICE=cuda
   ```

2. **Batch Processing**

   ```python
   # Increase batch size
   BATCH_SIZE = 32
   ```

3. **Caching**
   ```python
   # Add Redis caching
   REDIS_URL = "redis://localhost:6379"
   ```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](link-to-issues)
- **Email**: support@example.com
- **Documentation**: [Wiki](link-to-wiki)
- **Discord**: [Community](link-to-discord)

## 🙏 Acknowledgments

- [CLIP](https://github.com/openai/CLIP) - OpenAI's CLIP model
- [BEIT-3](https://github.com/microsoft/unilm/tree/master/beit3) - Microsoft's BEIT-3 model
- [Milvus](https://milvus.io/) - Vector database
- [Elasticsearch](https://www.elastic.co/) - Search engine
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
