# app/main.py
from fastapi import FastAPI, UploadFile, File
import os, shutil

from app.agents.document_agent import document_understanding_agent
from app.agents.retrieval_agent import retrieve_legal_context
from app.agents.answer_agent import answer_with_rag

app = FastAPI()

UPLOAD_DIR = "/tmp/legal_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze_with_rag")
async def analyze_with_rag(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    case_data = document_understanding_agent(file_path)
    retrieval = retrieve_legal_context(case_data)
    answer = answer_with_rag(case_data, retrieval)

    return {
        "case_data": case_data,
        "retrieved_documents": retrieval["documents"],
        "answer": answer
    }