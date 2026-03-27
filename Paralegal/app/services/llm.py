import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.services.parsing import parse_model_json

MODEL_NAME = os.getenv("KRIKRI_MODEL", "ilsp/Llama-Krikri-8B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def build_prompt(text_chunk: str) -> str:
    return f"""
Είσαι νομικός βοηθός για ελληνικά νομικά έγγραφα.

Ανάλυσε το παρακάτω κείμενο και επέστρεψε ΜΟΝΟ έγκυρο JSON με ακριβώς αυτά τα πεδία:

{{
  "summary": "σύντομη περίληψη",
  "parties": ["διάδικος 1", "διάδικος 2"],
  "legal_issues": ["ζήτημα 1", "ζήτημα 2"],
  "dates": ["ημερομηνία 1"],
  "key_arguments": ["επιχείρημα 1"],
  "citations": ["νόμος/απόφαση 1"]
}}

Κανόνες:
- Μην προσθέτεις εξηγήσεις εκτός JSON.
- Αν κάτι δεν υπάρχει, βάλε κενή λίστα ή κενό string.
- Μην επινοείς πληροφορίες.

Κείμενο:
{text_chunk}
""".strip()

def call_llm(text_chunk: str, max_new_tokens: int = 700):
    prompt = build_prompt(text_chunk)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id
        )

    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Keep only newly generated tail when possible
    generated_text = raw_text[len(prompt):].strip() if raw_text.startswith(prompt) else raw_text

    return parse_model_json(generated_text)