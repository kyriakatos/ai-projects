from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary: str
    input_length: int
```

---

### `requirements.txt`
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
transformers>=4.40.0
accelerate>=0.29.0
torch>=2.2.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
sentencepiece
protobuf