# --- START OF FILE app/ai_models/embedding_models.py ---

import logging
import torch
import numpy as np
from typing import Optional

# AI model imports
import open_clip
import sentencepiece as spm
from torchvision import transforms
from modeling_finetune import BEiT3ForRetrieval

from .beit3_config import BEiT3Config
from ..config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages CLIP and BEiT-3 models and embeddings"""
    
    def __init__(self, device: str):
        self.device = device
        self.initialized = False
        
        # Model placeholders
        self.clip_model: Optional[torch.nn.Module] = None
        self.clip_tokenizer = None
        self.beit3_model: Optional[torch.nn.Module] = None
        self.beit3_sp_model: Optional[spm.SentencePieceProcessor] = None
        
    def load_models(self):
        """Load all AI models onto the specified device"""
        if self.initialized:
            return
            
        logger.info(f"Loading AI models onto device: '{self.device}'")
        
        self._load_clip()
        self._load_beit3()
        
        self.initialized = True
        logger.info("✅ All embedding models loaded successfully.")
    
    def _load_clip(self):
        """Load CLIP model"""
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name='ViT-H-14', 
            pretrained=settings.CLIP_MODEL_PATH, 
            device=self.device
        )
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        logger.info("  - CLIP model loaded.")
    
    def _load_beit3(self):
        """Load BEiT-3 model"""
        self.beit3_model = BEiT3ForRetrieval(BEiT3Config())
        checkpoint = torch.load(settings.BEIT3_MODEL_PATH, map_location="cpu")
        self.beit3_model.load_state_dict(checkpoint["model"])
        self.beit3_model = self.beit3_model.to(self.device).eval()
        
        self.beit3_sp_model = spm.SentencePieceProcessor()
        self.beit3_sp_model.load(settings.BEIT3_SPM_PATH)
        
        logger.info("  - BEiT-3 model loaded.")
    
    def get_clip_text_embedding(self, text: str) -> np.ndarray:
        """Generate CLIP text embedding"""
        if not self.initialized or self.clip_model is None:
            raise RuntimeError("CLIP model is not loaded")
            
        with torch.no_grad():
            tokens = self.clip_tokenizer([text]).to(self.device)
            text_emb = self.clip_model.encode_text(tokens).cpu().numpy()[0]
            return text_emb / np.linalg.norm(text_emb, axis=0)
    
    def get_beit3_text_embedding(self, text: str) -> np.ndarray:
        """Generate BEiT-3 text embedding"""
        if not self.initialized or self.beit3_model is None:
            raise RuntimeError("BEiT-3 model is not loaded")
            
        with torch.no_grad():
            text_ids = self.beit3_sp_model.encode_as_ids(text)
            text_padding_mask = [0] * len(text_ids)
            text_ids_tensor = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            text_padding_mask_tensor = torch.tensor(text_padding_mask, dtype=torch.long).unsqueeze(0).to(self.device)
            _, text_emb = self.beit3_model(
                text_description=text_ids_tensor,
                text_padding_mask=text_padding_mask_tensor,
                only_infer=True
            )
            return text_emb.cpu().numpy()[0]

# --- END OF FILE app/ai_models/embedding_models.py ---
