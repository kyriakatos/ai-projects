/models/meltemi-7b-instruct.Q4_K_M.gguf

python summarize_greek_docx_meltemi.py \
  --model /models/meltemi-7b-instruct.Q4_K_M.gguf \
  --input /path/to/input.docx \
  --output /path/to/summary.docx \
  --output-txt /path/to/summary.txt

o run your Greek DOCX summarizer with the Meltemi LLM on Apple Silicon using Metal GPU acceleration, 



pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir


Then, modify the model loading section of your Python script to offload all layers to the GPU by setting n_gpu_layers=-1, enable a larger batch size with n_batch=1024, turn on memory mapping (use_mmap=True), and use half-precision KV cache (f16_kv=True). Keep your context window at n_ctx=8192, or increase it to 12288 if your Mac has 32GB RAM or more.

Use a Metal-friendly Meltemi GGUF model such as Q4_K_M for speed or Q5_K_M for better quality, e.g. meltemi-7b-instruct.Q5_K_M.gguf. Run the script with your model and DOCX input to generate the Greek one-page summary. When Metal is active, the console will show ggml_metal_init: using MPS, and you should see a significant speedup (often 5–10× faster) compared to CPU-only execution on M-series Macs.