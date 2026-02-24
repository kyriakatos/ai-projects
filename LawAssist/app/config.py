from pydantic import BaseModel
import os

class Settings(BaseModel):
    # Model
    model_path: str = os.getenv("MODEL_PATH", "models/model.gguf")
    n_ctx: int = int(os.getenv("N_CTX", "8192"))
    n_threads: int = int(os.getenv("N_THREADS", "6"))
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "50"))  # Apple Silicon Metal offload

    # Summarization defaults
    max_tokens: int = int(os.getenv("MAX_TOKENS", "700"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))

    # RAG
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "docs")

settings = Settings()