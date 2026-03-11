from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    logger.info(f"Loading model: {settings.model_name}")
    kwargs = {"token": settings.hf_token} if settings.hf_token else {}

    _tokenizer = AutoTokenizer.from_pretrained(settings.model_name, **kwargs)
    _model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        torch_dtype=torch.float16 if settings.device == "cuda" else torch.float32,
        device_map="auto",
        **kwargs,
    )
    logger.info("Model loaded successfully.")
    return _model, _tokenizer