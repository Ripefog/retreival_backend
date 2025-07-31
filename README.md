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
# Milvus Configuration (via ngrok)
MILVUS_HOST=0.tcp.ap.ngrok.io
MILVUS_PORT=13216
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

### Development mode
```bash
cd app
python main.py
```

### Production mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (tùy chọn)
```bash
docker build -t video-retrieval-backend .
docker run -p 8000:8000 video-retrieval-backend
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

## 🔧 Cấu hình

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MILVUS_HOST` | Milvus host (via ngrok) | `0.tcp.ap.ngrok.io` |
| `MILVUS_PORT` | Milvus port | `13216` |
| `ELASTICSEARCH_HOST` | Elasticsearch host | `0.tcp.ap.ngrok.io` |
| `ELASTICSEARCH_PORT` | Elasticsearch port | `9200` |
| `DEVICE` | Device for models | `cuda` |
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |

### Collections Configuration

| Collection | Purpose | Description |
|------------|---------|-------------|
| `arch_clip_image_v3` | CLIP embeddings | Vector embeddings từ CLIP model |
| `arch_beit3_image_v3` | BEIT-3 embeddings | Vector embeddings từ BEIT-3 model |
| `arch_object_name_v3` | Object detection | Object labels và bounding boxes |
| `arch_color_name_v3` | Color detection | Color information |

### Indices Configuration

| Index | Purpose | Description |
|-------|---------|-------------|
| `video_retrieval_metadata_v3` | Video metadata | Thông tin metadata của videos |
| `ocr` | OCR text | Text extracted từ images |
| `video_transcripts` | ASR transcripts | Speech transcripts |

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

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_retrieval.py::test_search
```

## 🐛 Troubleshooting

### Common Issues

1. **Milvus Connection Failed**
   ```bash
   # Check ngrok tunnel
   curl http://localhost:4040/api/tunnels
   
   # Check Milvus health
   curl http://localhost:8000/health
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