from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "ilsp/Llama-Krikri-8B-Base"
    max_new_tokens: int = 512
    max_input_chars: int = 8000
    chunk_size: int = 2000
    chunk_overlap: int = 200
    device: str = "cuda"  # or "cpu"
    hf_token: str = ""    # optional, for gated models

    class Config:
        env_file = ".env"

settings = Settings()