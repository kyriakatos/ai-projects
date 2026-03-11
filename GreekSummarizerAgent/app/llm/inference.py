import torch
from app.llm.loader import load_model
from app.utils.config import settings

def generate_summary(prompt: str) -> str:
    model, tokenizer = load_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Strip the prompt from output (base model returns full sequence)
    return decoded[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()