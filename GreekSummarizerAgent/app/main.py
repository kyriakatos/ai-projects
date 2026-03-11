from fastapi import FastAPI, HTTPException
from app.schemas.models import SummarizeRequest, SummarizeResponse
from app.agent import SummarizerAgent
from app.llm.loader import load_model

app = FastAPI(title="Greek Summarizer Agent")
agent = SummarizerAgent()

@app.on_event("startup")
async def startup():
    load_model()  # Warm up model on start

@app.post("/summarize", response_model=SummarizeResponse)
def summarize_document(request: SummarizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty document provided.")
    result = agent.run(request.text)
    return SummarizeResponse(**result)

@app.get("/health")
def health():
    return {"status": "ok"}