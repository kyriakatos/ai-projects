##This makes it Azure-friendly ( HTTP endpoint)

from fastapi import FastAPI
from pydantic import BaseModel
from agents import Runner

from .agent import build_agent
from .logging_config import setup_logging

setup_logging()

app = FastAPI(title="Stock Agent API")
agent = build_agent()

class RunRequest(BaseModel):
    prompt: str

@app.post("/run")
def run_agent(req: RunRequest):
    result = Runner.run_sync(agent, req.prompt)
    return {"output": result.final_output}


## Run locally from bash 
## uvicorn stock_agent.api:app --host 0.0.0.0 --port 8000

