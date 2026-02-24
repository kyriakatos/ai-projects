from llama_cpp import Llama
from ..config import settings
from .prompts import system_prompt_gr, build_user_prompt
from typing import Optional, Dict, Any, List
import time
import re

_llm: Optional[Llama] = None

def get_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=settings.model_path,
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            n_gpu_layers=settings.n_gpu_layers,  # Metal offload on Apple Silicon
            verbose=False
        )
    return _llm

def _extract_sections(text: str) -> Dict[str, Any]:
    """
    Very simple parser for:
    Περίληψη:
    Βασικά σημεία:
    Ενέργειες:
    """
    def grab(section: str) -> str:
        m = re.search(rf"{section}:\s*(.*?)(\n[A-ΩΆΈΉΊΌΎΏA-Za-z ].*?:\s*|\Z)", text, flags=re.S)
        return (m.group(1).strip() if m else "").strip()

    summary = grab("Περίληψη")
    points = grab("Βασικά σημεία")
    actions = grab("Ενέργειες")

    def bullets_to_list(block: str) -> Optional[List[str]]:
        if not block:
            return None
        lines = [re.sub(r"^\s*[-•]\s*", "", ln).strip() for ln in block.splitlines()]
        lines = [ln for ln in lines if ln]
        return lines or None

    return {
        "summary": summary or text.strip(),
        "key_points": bullets_to_list(points),
        "action_items": bullets_to_list(actions),
    }

def summarize_greek(
    text: str,
    style: str = "executive",
    bullets: bool = True,
    include_actions: bool = True,
    max_words: Optional[int] = None,
    rag_context: Optional[str] = None,
) -> Dict[str, Any]:
    llm = get_llm()
    sys = system_prompt_gr()
    user = build_user_prompt(text=text, style=style, bullets=bullets, include_actions=include_actions, max_words=max_words, rag_context=rag_context)

    # Chat format: use simple chat completion pattern
    start = time.time()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )
    elapsed = time.time() - start
    content = out["choices"][0]["message"]["content"]

    sections = _extract_sections(content)
    meta = {"latency_s": round(elapsed, 3), "model_path": settings.model_path}

    return {"language": "el", **sections, "raw": content, "meta": meta}