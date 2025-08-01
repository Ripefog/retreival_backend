# --------------------------------------------------------------------------------
# # import
# --------------------------------------------------------------------------------

!pip install virtualenv

!virtualenv -p python3.8 venv
!source venv/bin/activate && python --version

!source venv/bin/activate && pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
!source venv/bin/activate && python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
!source venv/bin/activate && pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

print("Downloading CLIP model in open_clip-compatible .bin format...")
!wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin -O /kaggle/working/clip_model.bin
print("Download complete.")

# Cài Co-DETR
!source venv/bin/activate && git clone https://github.com/Sense-X/Co-DETR.git
!source venv/bin/activate && pip install -e /kaggle/working/Co-DETR
!source venv/bin/activate && mv /kaggle/working/Co-DETR /kaggle/working/Co_DETR

!source venv/bin/activate && pip uninstall -y timm
!source venv/bin/activate && pip install timm==0.4.12

!source venv/bin/activate && pip install scikit-learn

# Cài đặt các thư viện hỗ trợ với virtualenv venv
!source venv/bin/activate && pip install \
    moviepy && \
source venv/bin/activate && pip install \
    opensearch-py && \
source venv/bin/activate && pip install \
    requests-aws4auth && \
source venv/bin/activate && pip install \
    boto3 && \
source venv/bin/activate && pip install \
    nbimporter && \
source venv/bin/activate && pip install \
    transformers && \
source venv/bin/activate && pip install \
    torch && \
source venv/bin/activate && pip install \
    pillow && \
source venv/bin/activate && pip install \
    open-clip-torch && \
source venv/bin/activate && pip install \
    pymilvus && \
source venv/bin/activate && pip install \
    lmdb && \
source venv/bin/activate && pip install \
    nbformat && \
source venv/bin/activate && pip install \
    ipython

!source venv/bin/activate && pip install "setuptools<58.0.0"

!source venv/bin/activate && pip install opensearch-py requests-aws4auth boto3

!source venv/bin/activate && git clone https://github.com/microsoft/unilm.git
# !source venv/bin/activate && pip install torch torchvision
!source venv/bin/activate && pip install -r /kaggle/working/unilm/beit3/requirements.txt
!source venv/bin/activate && pip uninstall protobuf -y && pip install protobuf==3.20.3
!wget https://github.com/addf400/files/releases/download/beit3/beit3.spm
!wget https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth

# --------------------------------------------------------------------------------
# # check_versions.py
# --------------------------------------------------------------------------------

%%writefile /kaggle/working/check_versions.py
import sys
import torch
import subprocess

def check_versions():
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    
    # Python version
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # GPU information
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name,driver_version,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            print(f"GPU: {gpu_info[0]}")
            print(f"Driver version: {gpu_info[1]}")
            print(f"GPU memory: {gpu_info[2]} MB")
    except:
        print("Could not retrieve GPU information")
    
    print("=" * 50)

# Chạy kiểm tra
check_versions()

!source venv/bin/activate && python check_versions.py

import os
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# Initialize the secrets client
user_secrets = UserSecretsClient()

# --- OpenSearch Secrets ---
#OPENSEARCH_HOST = user_secrets.get_secret("OPENSEARCH_HOST")
#OPENSEARCH_USERNAME = user_secrets.get_secret("OPENSEARCH_USERNAME")
#OPENSEARCH_PASSWORD = user_secrets.get_secret("OPENSEARCH_PASSWORD")

# --- Milvus Secrets (set as environment variables) ---
os.environ["MILVUS_URI"] = user_secrets.get_secret("MILVUS_URI")
os.environ["MILVUS_TOKEN"] = user_secrets.get_secret("MILVUS_TOKEN")
os.environ["OPENSEARCH_HOST"] = user_secrets.get_secret("OPENSEARCH_HOST")
os.environ["OPENSEARCH_USERNAME"] = user_secrets.get_secret("OPENSEARCH_USERNAME")
os.environ["OPENSEARCH_PASSWORD"] = user_secrets.get_secret("OPENSEARCH_PASSWORD")
# --- Hugging Face Secret (set as environment variable) ---
# We get the secret you created named "HF_TOKEN"
# and set it to the standard environment variable name that Hugging Face libraries look for.
# os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
# hf_token = os.environ["HF_TOKEN"]
# --- Login and Verification ---
print(f"MILVUS_URI: {os.environ['MILVUS_URI']}")
# IMPORTANT: Avoid printing the actual tokens for security.
print("MILVUS_TOKEN loaded.") 
print("All secrets loaded.")

# # Log in to Hugging Face Hub using the token from the environment variable
# try:
#     login()
#     print("Successfully logged in to Hugging Face Hub.")
# except Exception as e:
#     print(f"Could not log in to Hugging Face Hub. Error: {e}")

# --------------------------------------------------------------------------------
# # config.py
# --------------------------------------------------------------------------------

%%writefile /kaggle/working/config.py
import os

def check_env(var_name):
    value = os.environ.get(var_name)
    if value is None:
        print(f"[❌] Environment variable '{var_name}' is NOT set.")
    else:
        print(f"[✅] {var_name} = {value}")
    return value

# --- CONFIG CHO PIPELINE INDEX DỮ LIỆU ---
class Config_Indexing:
    DEVICE = "cuda" if "cuda" in os.popen("nvidia-smi").read().lower() else "cpu"
    KEYFRAME_ROOT_DIR = "/kaggle/input/aloaic/keyframes_output" 
    
    MILVUS_URI = check_env("MILVUS_URI")
    MILVUS_TOKEN = check_env("MILVUS_TOKEN")
    
    CLIP_COLLECTION = 'arch_clip_image_v3'
    BEIT3_COLLECTION = 'arch_beit3_image_v3'
    OBJECT_COLLECTION = 'arch_object_name_v3'
    COLOR_COLLECTION = 'arch_color_name_v3'
    
    OPENSEARCH_HOST = check_env("OPENSEARCH_HOST")
    OPENSEARCH_USERNAME = check_env("OPENSEARCH_USERNAME")
    OPENSEARCH_PASSWORD = check_env("OPENSEARCH_PASSWORD")
    
    METADATA_INDEX = 'video_retrieval_metadata_v3'

# --- CONFIG CHO MODULE TRUY XUẤT (RETRIEVAL) ---
class Config_Retrieval:
    DEVICE = "cuda" if "cuda" in os.popen("nvidia-smi").read().lower() else "cpu"
    
    MILVUS_URI = check_env("MILVUS_URI")
    MILVUS_TOKEN = check_env("MILVUS_TOKEN")
    
    OPENSEARCH_HOST = check_env("OPENSEARCH_HOST")
    OPENSEARCH_USERNAME = check_env("OPENSEARCH_USERNAME")
    OPENSEARCH_PASSWORD = check_env("OPENSEARCH_PASSWORD")
    
    CLIP_COLLECTION = 'arch_clip_image_v3'
    BEIT3_COLLECTION = 'arch_beit3_image_v3'
    OBJECT_COLLECTION = 'arch_object_name_v3'
    COLOR_COLLECTION = 'arch_color_name_v3'
    METADATA_INDEX = 'video_retrieval_metadata_v3'

    OCR_INDEX = 'ocr'
    ASR_INDEX = 'video_transcripts'

print("[✔] Configuration file loaded.")


!source venv/bin/activate && python config.py

# --------------------------------------------------------------------------------
# # process.py
# --------------------------------------------------------------------------------

%%writefile /kaggle/working/process.py
import cv2
import torch
import numpy as np
import open_clip
from PIL import Image
from typing import Tuple
from transformers import AutoImageProcessor, BeitModel
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import os
from mmcv import Config
import sys
sys.path.append('/kaggle/working/Co_DETR')
from mmdet.apis import init_detector, inference_detector
import sys
sys.path.append('/kaggle/working/unilm/beit3')
from modeling_finetune import BEiT3ForRetrieval
from torchvision import transforms
from torchscale.architecture.config import *

# PROPER IMPORT cho BEiT-3 tokenizer
import sentencepiece as spm

# --- Cấu hình cho BEiT-3 (giữ nguyên) ---
class BEiT3Config:
    def __init__(self):
        self.encoder_embed_dim = 768
        self.encoder_attention_heads = 12
        self.encoder_layers = 12
        self.encoder_ffn_embed_dim = 3072
        self.img_size = 384
        self.patch_size = 16
        self.in_chans = 3
        self.vocab_size = 64010
        self.num_max_bpe_tokens = 64
        self.max_source_positions = 1024
        self.multiway = True
        self.share_encoder_input_output_embed = False
        self.no_scale_embedding = False
        self.layernorm_embedding = False
        self.normalize_output = True
        self.no_output_layer = True
        self.drop_path_rate = 0.1
        self.dropout = 0.0
        self.attention_dropout = 0.0
        self.drop_path = 0.1
        self.activation_dropout = 0.0
        self.max_position_embeddings = 1024
        self.encoder_normalize_before = True
        self.activation_fn = "gelu"
        self.encoder_learned_pos = True
        self.xpos_rel_pos = False
        self.xpos_scale_base = 512
        self.checkpoint_activations = False
        self.deepnorm = False
        self.subln = True
        self.rel_pos_buckets = 0
        self.max_rel_pos = 0
        self.bert_init = False
        self.moe_freq = 0
        self.moe_expert_count = 0
        self.moe_top1_expert = False
        self.moe_gating_use_fp32 = True
        self.moe_eval_capacity_token_fraction = 0.25
        self.moe_second_expert_policy = "random"
        self.moe_normalize_gate_prob_before_dropping = False
        self.use_xmoe = False
        self.fsdp = False
        self.ddp_rank = 0
        self.flash_attention = False
        self.scale_length = 2048
        self.layernorm_eps = 1e-5

# --- Lớp trích xuất đặc trưng chính (FIXED) ---
class FeatureExtractor:
    def __init__(self, device):
        print("Initializing unified FeatureExtractor...")
        self.device = device
        self.beit3_loaded = False

        # === 1. Tải mô hình CLIP (giữ nguyên) ===
        print("Loading CLIP model (ViT-H-14)...")
        clip_model_name = 'ViT-H-14'
        clip_model_path = '/kaggle/working/clip_model.bin'
        if not os.path.exists(clip_model_path):
            raise FileNotFoundError(f"CLIP model not found at {clip_model_path}.")

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name=clip_model_name,
            pretrained=clip_model_path,
            device=self.device
        )
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        print("[✔] CLIP model loaded.")

        # === 2. Tải mô hình BEiT-3 (PROPER WAY) ===
        print("Loading BEiT-3 model...")
        try:
            beit3_checkpoint_path = "/kaggle/working/beit3_base_patch16_384_coco_retrieval.pth"
            if not os.path.exists(beit3_checkpoint_path):
                raise FileNotFoundError(f"BEiT-3 model not found.")

            beit3_config = BEiT3Config()
            self.beit3_model = BEiT3ForRetrieval(beit3_config)
            checkpoint = torch.load(beit3_checkpoint_path, map_location="cpu")
            self.beit3_model.load_state_dict(checkpoint["model"])
            self.beit3_model = self.beit3_model.to(self.device).eval()

            # PROPER TOKENIZER LOADING
            print("Loading BEiT-3 tokenizer (SentencePiece)...")
            spm_path = "/kaggle/working/beit3.spm"
            if not os.path.exists(spm_path):
                raise FileNotFoundError(f"Tokenizer not found at {spm_path}")
            
            self.beit3_sp_model = spm.SentencePieceProcessor()
            self.beit3_sp_model.load(spm_path)

            self.beit3_preprocess = transforms.Compose([
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            
            self.beit3_loaded = True
            print("[✔] BEiT-3 model and tokenizer loaded successfully.")
            
        except Exception as e:
            print(f"[❌] BEiT-3 loading failed: {e}")
            self.beit3_loaded = False

        print(f"\n[✔] FeatureExtractor ready - CLIP: ✅, BEiT-3: {'✅' if self.beit3_loaded else '❌'}")

    def get_image_embeddings(self, frame_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Trích xuất embedding cho ảnh."""
        image_pil = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
        
        with torch.no_grad():
            # CLIP
            clip_image = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
            clip_emb = self.clip_model.encode_image(clip_image).cpu().numpy()[0]
            clip_emb /= np.linalg.norm(clip_emb)

            # BEiT-3
            if self.beit3_loaded:
                try:
                    beit3_image = self.beit3_preprocess(image_pil).unsqueeze(0).to(self.device)
                    beit3_emb, _ = self.beit3_model(image=beit3_image, only_infer=True)
                    beit3_emb = beit3_emb.cpu().numpy()[0]
                except Exception as e:
                    print(f"BEiT-3 image encoding error: {e}")
                    beit3_emb = np.zeros(768)
            else:
                beit3_emb = np.zeros(768)

        return clip_emb, beit3_emb

    def get_clip_text_embedding(self, text: str) -> np.ndarray:
        """CLIP text embedding."""
        with torch.no_grad():
            tokens = self.clip_tokenizer([text]).to(self.device)
            text_emb = self.clip_model.encode_text(tokens).cpu().numpy()[0]
            text_emb /= np.linalg.norm(text_emb)
        return text_emb

    def get_beit3_text_embedding(self, text: str) -> np.ndarray:
        """BEiT-3 text embedding (PROPER WAY)."""
        if not self.beit3_loaded:
            return np.zeros(768)
            
        try:
            with torch.no_grad():
                # PROPER TOKENIZATION
                text_ids = self.beit3_sp_model.encode_as_ids(text)
                text_padding_mask = [0] * len(text_ids)

                text_ids = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                text_padding_mask = torch.tensor(text_padding_mask, dtype=torch.long).unsqueeze(0).to(self.device)

                _, text_emb = self.beit3_model(
                    text_description=text_ids,
                    text_padding_mask=text_padding_mask,
                    only_infer=True
                )
                return text_emb.cpu().numpy()[0]
        except Exception as e:
            print(f"BEiT-3 text encoding error: {e}")
            return np.zeros(768)

# --- ObjectColorDetector (giữ nguyên) ---
class ObjectColorDetector:
    """Sử dụng Co-DETR để phát hiện đối tượng và màu sắc."""
    def __init__(self, device):
        print("\nInitializing ObjectColorDetector (Co-DETR)...")
        self.device = device
        config_path = "/kaggle/working/Co_DETR/projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py"
        ckpt_path = "/kaggle/input/alockpt/co_dino_5scale_swin_large_16e_o365tococo.pth"

        sys.path.append('/kaggle/working/Co_DETR')
        self.model = init_detector(Config.fromfile(config_path), ckpt_path, device=self.device)

        self.basic_colors = {'red':(255,0,0),'green':(0,255,0),'blue':(0,0,255),'yellow':(255,255,0),'cyan':(0,255,255),'magenta':(255,0,255),'black':(0,0,0),'white':(255,255,255),'gray':(128,128,128),'orange':(255,165,0),'brown':(165,42,42),'pink':(255,192,203), 'purple': (128,0,128)}
        self.color_names = list(self.basic_colors.keys())
        self.color_tree = KDTree(np.array(list(self.basic_colors.values())))
        print("[✔] Co-DETR model loaded.")

    def detect(self, image_path: str) -> Tuple[list, list]:
        try:
            from mmdet.apis import inference_detector
            object_labels, color_labels = [], []
            result = inference_detector(self.model, image_path)
            if isinstance(result, tuple):
                result = result[0]

            frame_cv = cv2.imread(image_path)
            if frame_cv is None:
                return [], []

            for class_id, bboxes in enumerate(result):
                if class_id >= len(self.model.CLASSES): continue
                for bbox in bboxes:
                    if bbox[4] < 0.5: continue
                    object_labels.append(self.model.CLASSES[class_id])

                    x1, y1, x2, y2 = map(int, bbox[:4])
                    crop = frame_cv[y1:y2, x1:x2]
                    if crop.size == 0: continue

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3)
                    kmeans = KMeans(n_clusters=1, n_init=10, random_state=0).fit(crop_rgb)
                    _, idx = self.color_tree.query([kmeans.cluster_centers_[0]], k=1)
                    color_labels.append(self.color_names[idx[0][0]])

            return list(set(object_labels)), list(set(color_labels))
        except Exception as e:
            print(f"Object/color detection error: {e}")
            return [], []

!source venv/bin/activate && python process.py

# --------------------------------------------------------------------------------
# # retriever.py
# --------------------------------------------------------------------------------

%%writefile /kaggle/working/retriever.py
import torch
from pymilvus import Collection, connections
from opensearchpy import OpenSearch, NotFoundError
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

# Import các module đã tạo
from process import FeatureExtractor
from config import Config_Retrieval

class HybridRetriever:
    """
    Một lớp retriever lai, triển khai logic "lọc và tinh chỉnh" như mô tả trong paper.
    - Mode 'clip'/'beit3': Tìm kiếm chỉ bằng một mô hình.
    - Mode 'hybrid': Dùng CLIP để lọc rộng, sau đó dùng BEIT-3 để tinh chỉnh và xếp hạng lại.
    """
    def __init__(self, config: Config_Retrieval):
        print("Initializing Hybrid Retriever...")
        self.config = config
        self.device = config.DEVICE

        # 1. Khởi tạo FeatureExtractor để mã hóa truy vấn
        self.feature_extractor = FeatureExtractor(self.device)
        print("[✔] Feature Extractor for queries initialized.")

        # 2. Kết nối tới Milvus và tải các collection
        connections.connect("retrieval_alias", uri=config.MILVUS_URI, token=config.MILVUS_TOKEN, secure=True)
        self.collections = {
            'clip': Collection(config.CLIP_COLLECTION, using="retrieval_alias"),
            'beit3': Collection(config.BEIT3_COLLECTION, using="retrieval_alias"),
            'object': Collection(config.OBJECT_COLLECTION, using="retrieval_alias"),
            'color': Collection(config.COLOR_COLLECTION, using="retrieval_alias"),
        }
        for name, coll in self.collections.items():
            coll.load()
            print(f"[✔] Milvus collection '{name}' loaded.")

        # 3. Kết nối tới Elasticsearch/OpenSearch
        self.es_client = OpenSearch(
            hosts=[{'host': config.OPENSEARCH_HOST, 'port': 443}],
            http_auth=(config.OPENSEARCH_USERNAME, config.OPENSEARCH_PASSWORD),
            use_ssl=True, verify_certs=True,
            ssl_assert_hostname=False, ssl_show_warn=False,
        )
        if self.es_client.ping():
            print("[✔] Successfully connected to OpenSearch.")
        else:
            raise ConnectionError("Could not connect to OpenSearch.")

    def _search_milvus(self, collection_name: str, vector: List[float], top_k: int) -> List[Dict]:
        """Hàm helper để tìm kiếm trên một collection Milvus."""
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
        
        output_fields = ["keyframe_id", "video_id", "timestamp"]
        if any(field.name == "label" for field in self.collections[collection_name].schema.fields):
            output_fields.append("label")
            
        results = self.collections[collection_name].search(
            data=[vector], anns_field="vector", param=search_params,
            limit=top_k, output_fields=output_fields
        )
        
        hits = []
        for hit in results[0]:
            hits.append({
                'id': hit.id,
                'distance': hit.distance,
                'entity': hit.entity.to_dict()
            })
        return hits

    def _search_es(self, ocr_query: Optional[str], asr_query: Optional[str]) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """Tìm kiếm trên OpenSearch để lấy ID cho việc lọc."""
        ocr_kf_ids, asr_video_ids = None, None

        if ocr_query:
            ocr_kf_ids = set()
            query_body = {"query": {"match": {"text": ocr_query}}, "_source": ["keyframe_id"]}
            try:
                res = self.es_client.search(index=self.config.OCR_INDEX, body=query_body, size=10000)
                for hit in res['hits']['hits']:
                    ocr_kf_ids.add(hit['_source']['keyframe_id'])
            except NotFoundError:
                print(f"Warning: OCR index '{self.config.OCR_INDEX}' not found.")

        if asr_query:
            asr_video_ids = set()
            query_body = {"query": {"match": {"text": asr_query}}, "_source": ["video_id"]}
            try:
                res = self.es_client.search(index=self.config.ASR_INDEX, body=query_body, size=10000)
                for hit in res['hits']['hits']:
                    asr_video_ids.add(hit['_source']['video_id'])
            except NotFoundError:
                print(f"Warning: ASR index '{self.config.ASR_INDEX}' not found.")
        
        return ocr_kf_ids, asr_video_ids

    def search(
        self,
        text_query: str,
        mode: str = 'hybrid',
        object_filters: Optional[List[str]] = None,
        color_filters: Optional[List[str]] = None,
        ocr_query: Optional[str] = None,
        asr_query: Optional[str] = None,
        top_k: int = 100
    ) -> List[Dict]:
        if mode not in ['hybrid', 'clip', 'beit3']:
            raise ValueError("Mode must be one of 'hybrid', 'clip', or 'beit3'")
            
        print(f"\n--- [STARTING SEARCH] Query: '{text_query}', Mode: {mode.upper()} ---")
        
        num_initial_candidates = top_k * 5
        candidate_info = {}

        # GIAI ĐOẠN 1: LẤY ỨNG VIÊN BAN ĐẦU
        if mode == 'clip' or mode == 'hybrid':
            print("1. Broad Semantic Filtering with CLIP...")
            clip_vector = self.feature_extractor.get_clip_text_embedding(text_query).tolist()
            clip_candidates = self._search_milvus('clip', clip_vector, num_initial_candidates)
            print(f"   - Found {len(clip_candidates)} initial candidates from CLIP.")
            
            for hit in clip_candidates:
                kf_id = hit['entity']['keyframe_id']
                score = 1.0 / (1.0 + hit['distance'])
                candidate_info[kf_id] = {
                    'video_id': hit['entity']['video_id'],
                    'timestamp': hit['entity']['timestamp'],
                    'clip_score': score,
                    'score': score, # Điểm tổng hợp ban đầu
                    'reasons': [f"CLIP match (score: {score:.3f})"]
                }
        
        elif mode == 'beit3':
            print("1. Searching with BEIT-3 only...")
            beit3_vector = self.feature_extractor.get_beit3_text_embedding(text_query).tolist()
            beit3_candidates = self._search_milvus('beit3', beit3_vector, num_initial_candidates)
            print(f"   - Found {len(beit3_candidates)} candidates from BEIT-3.")
            
            for hit in beit3_candidates:
                kf_id = hit['entity']['keyframe_id']
                score = 1.0 / (1.0 + hit['distance'])
                candidate_info[kf_id] = {
                    'video_id': hit['entity']['video_id'],
                    'timestamp': hit['entity']['timestamp'],
                    'beit3_score': score,
                    'score': score,
                    'reasons': [f"BEIT-3 match (score: {score:.3f})"]
                }
        
        # GIAI ĐOẠN 2: TINH CHỈNH BẰNG BEIT-3 (CHỈ CHO MODE HYBRID)
        if mode == 'hybrid' and candidate_info:
            print("2. Refining results with BEIT-3...")
            candidate_kf_ids = list(candidate_info.keys())
            
            # Sử dụng collection.query để lấy vector BEIT-3 của các ứng viên một cách hiệu quả
            expr = f'keyframe_id in {candidate_kf_ids}'
            try:
                beit3_vectors_of_candidates = self.collections['beit3'].query(
                    expr=expr,
                    output_fields=["keyframe_id", "vector"]
                )
                beit3_vector_map = {item['keyframe_id']: item['vector'] for item in beit3_vectors_of_candidates}
                
                beit3_query_vector = np.array(self.feature_extractor.get_beit3_text_embedding(text_query))
                
                for kf_id, info in candidate_info.items():
                    if kf_id in beit3_vector_map:
                        dist = np.linalg.norm(beit3_query_vector - np.array(beit3_vector_map[kf_id]))
                        beit3_score = 1.0 / (1.0 + dist)
                        
                        # Kết hợp điểm: 40% từ CLIP, 60% từ BEIT-3
                        combined_score = (0.4 * info['clip_score']) + (0.6 * beit3_score)
                        
                        info['score'] = combined_score
                        info['beit3_score'] = beit3_score
                        info['reasons'].append(f"BEIT-3 refine (score: {beit3_score:.3f})")
                    else:
                        info['score'] *= 0.8 # Hạ điểm nếu không có vector BEIT-3
                        info['reasons'].append("BEIT-3 vector not found for refinement")
                print(f"   - Re-ranked {len(candidate_info)} candidates.")
            except Exception as e:
                print(f"   - Error during BEIT-3 refinement: {e}. Skipping this step.")

        # GIAI ĐOẠN 3: TĂNG ĐIỂM (BOOSTING)
        print("3. Boosting scores with object/color filters...")
        # Sử dụng CLIP để mã hóa text của object/color vì nó mạnh hơn cho khái niệm đơn lẻ
        if object_filters:
            for obj in object_filters:
                obj_vector = self.feature_extractor.get_clip_text_embedding(obj).tolist()
                obj_hits = self._search_milvus('object', obj_vector, top_k * 10)
                for hit in obj_hits:
                    # Trích xuất keyframe_id gốc từ PK của entry object/color
                    kf_id = '_'.join(hit['id'].split('_')[:-2])
                    if kf_id in candidate_info:
                        candidate_info[kf_id]['score'] += 0.1 # Tăng điểm một lượng đáng kể
                        candidate_info[kf_id]['reasons'].append(f"Object match: '{obj}'")
        
        if color_filters:
            for color in color_filters:
                color_vector = self.feature_extractor.get_clip_text_embedding(color).tolist()
                color_hits = self._search_milvus('color', color_vector, top_k * 10)
                for hit in color_hits:
                    kf_id = '_'.join(hit['id'].split('_')[:-2])
                    if kf_id in candidate_info:
                        candidate_info[kf_id]['score'] += 0.05
                        candidate_info[kf_id]['reasons'].append(f"Color match: '{color}'")

        # GIAI ĐOẠN 4: LỌC CỨNG (HARD FILTERING)
        print("4. Hard filtering with OpenSearch (OCR/ASR)...")
        ocr_kf_ids, asr_video_ids = self._search_es(ocr_query, asr_query)
        
        if ocr_kf_ids is not None or asr_video_ids is not None:
            final_candidates = {}
            for kf_id, info in candidate_info.items():
                is_match = False
                if ocr_kf_ids is not None and kf_id in ocr_kf_ids:
                    info['score'] += 0.5 # Tăng điểm mạnh cho khớp OCR
                    info['reasons'].append("OCR text match")
                    is_match = True
                if asr_video_ids is not None and info['video_id'] in asr_video_ids:
                    info['score'] += 0.3 # Tăng điểm cho khớp ASR
                    info['reasons'].append("ASR transcript match")
                    is_match = True
                if is_match:
                    final_candidates[kf_id] = info
            candidate_info = final_candidates
            print(f"   - Remaining candidates after ES filtering: {len(candidate_info)}")
        else:
            print("   - No ES filters applied.")

        # GIAI ĐOẠN 5: XẾP HẠNG VÀ TRẢ VỀ
        print("5. Sorting and returning final results.")
        sorted_results = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        
        final_results = []
        for kf_id, info in sorted_results[:top_k]:
            # Xóa các điểm số phụ để output gọn gàng hơn
            info.pop('clip_score', None)
            info.pop('beit3_score', None)
            final_results.append({'keyframe_id': kf_id, **info})
            
        print(f"--- [SEARCH FINISHED] Returning {len(final_results)} results. ---")
        return final_results

!source venv/bin/activate && python retriever.py

# --------------------------------------------------------------------------------
# # main_retrieval.py
# --------------------------------------------------------------------------------

%%writefile /kaggle/working/main_retrieval.py
from retriever import HybridRetriever
from config import Config_Retrieval
import pprint

if __name__ == '__main__':
    # --- KHỞI TẠO HỆ THỐNG ---
    print("--- [INITIALIZING RETRIEVAL SYSTEM] ---")
    config = Config_Retrieval()
    retriever = HybridRetriever(config)
    print("--- [INITIALIZATION COMPLETE] ---")


    # --- TEST 1: SO SÁNH CÁC CHẾ ĐỘ TRUY XUẤT ---
    print("\n\n" + "="*60)
    print("      TEST 1: COMPARING RETRIEVAL MODES SIDE-BY-SIDE")
    print("="*60)
    
    test_query_1 = "a news anchor in a studio"
    top_k_results = 5

    print(f"\n--- [QUERY: '{test_query_1}', MODE: HYBRID] ---")
    results_hybrid = retriever.search(
        text_query=test_query_1,
        mode='hybrid',
        top_k=top_k_results
    )
    pprint.pprint(results_hybrid)

    print(f"\n--- [QUERY: '{test_query_1}', MODE: CLIP-ONLY] ---")
    results_clip = retriever.search(
        text_query=test_query_1,
        mode='clip',
        top_k=top_k_results
    )
    pprint.pprint(results_clip)

    print(f"\n--- [QUERY: '{test_query_1}', MODE: BEIT3-ONLY] ---")
    results_beit3 = retriever.search(
        text_query=test_query_1,
        mode='beit3',
        top_k=top_k_results
    )
    pprint.pprint(results_beit3)


    # --- TEST 2: TRUY VẤN PHỨC HỢP VỚI BỘ LỌC ---
    print("\n\n" + "="*60)
    print("      TEST 2: COMPLEX QUERY WITH ALL FILTERS (HYBRID MODE)")
    print("="*60)

    results_complex = retriever.search(
        text_query="a presenter on stage",
        mode='hybrid',
        object_filters=["person"],
        color_filters=["red"],
        ocr_query="VIỆT NAM",
        asr_query="kinh tế",
        top_k=5
    )
    pprint.pprint(results_complex)


    # --- TEST 3: TRUY VẤN TỪ KHÓA TIẾNG VIỆT ---
    print("\n\n" + "="*60)
    print("      TEST 3: SPECIFIC VIETNAMESE KEYWORD QUERY")
    print("="*60)
    
    test_query_3 = "Va chạm giao thông trên đường"
    print(f"\n--- [QUERY: '{test_query_3}', MODE: HYBRID] ---")
    results_3 = retriever.search(
        text_query=test_query_3,
        mode='hybrid',
        top_k=5
    )
    pprint.pprint(results_3)


    # --- TEST 4: TRUY VẤN MỤC TIÊU ---
    print("\n\n" + "="*60)
    print("      TEST 4: TARGETED RETRIEVAL FOR A SPECIFIC IMAGE")
    print("="*60)
    
    results_targeted = retriever.search(
        text_query="nữ biên tập viên mặc áo hồng đang dẫn chương trình thời sự",
        mode='hybrid',
        object_filters=["person"],
        color_filters=["pink", "red"],
        #ocr_query="60 giây",
        top_k=5
    )
    pprint.pprint(results_targeted)


    # --- TEST 5: TRUY VẤN NGỮ NGHĨA ĐƠN GIẢN ---
    print("\n\n" + "="*60)
    print("      TEST 5: SIMPLE SEMANTIC QUERY")
    print("="*60)

    test_query_5 = "a person sitting at a table"

    print(f"\n--- [QUERY: '{test_query_5}', MODE: HYBRID] ---")
    results_5a = retriever.search(
        text_query=test_query_5,
        mode='hybrid',
        top_k=5
    )
    pprint.pprint(results_5a)

    print(f"\n--- [QUERY: '{test_query_5}', MODE: CLIP] ---")
    results_5b = retriever.search(
        text_query=test_query_5,
        mode='clip',
        top_k=5
    )
    pprint.pprint(results_5b)

    print(f"\n--- [QUERY: '{test_query_5}', MODE: BEIT3] ---")
    results_5c = retriever.search(
        text_query=test_query_5,
        mode='beit3',
        top_k=5
    )
    pprint.pprint(results_5c)


    # --- TEST 6: TRUY VẤN PHỨC HỢP VỚI OBJECT + COLOR ---
    print("\n\n" + "="*60)
    print("      TEST 6: OBJECT + COLOR FILTER QUERY")
    print("="*60)

    test_query_6 = "a man wearing something blue"

    print(f"\n--- [QUERY: '{test_query_6}', MODE: HYBRID] ---")
    results_6a = retriever.search(
        text_query=test_query_6,
        mode='hybrid',
        object_filters=["person", "man"],
        color_filters=["blue"],
        top_k=5
    )
    pprint.pprint(results_6a)

    print(f"\n--- [QUERY: '{test_query_6}', MODE: CLIP] ---")
    results_6b = retriever.search(
        text_query=test_query_6,
        mode='clip',
        object_filters=["person", "man"],
        color_filters=["blue"],
        top_k=5
    )
    pprint.pprint(results_6b)

    print(f"\n--- [QUERY: '{test_query_6}', MODE: BEIT3] ---")
    results_6c = retriever.search(
        text_query=test_query_6,
        mode='beit3',
        object_filters=["person", "man"],
        color_filters=["blue"],
        top_k=5
    )
    pprint.pprint(results_6c)


!source venv/bin/activate && python main_retrieval.py


