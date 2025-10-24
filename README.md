### What is Aether?
Aether is a **monolithic, PyTorch-first runflow** for reproducible experiments:
- Single-file orchestration + config-driven runs
- Built-in logging & trace replay
- Optimizer stack: AdamW (decoupled weight decay), cosine/OneCycle schedulers,
  AMP/bfloat16, gradient clipping, EMA
- MPS acceleration (Apple Silicon) with torch.compile and alignment padding

> This is a **Public Spec build**. Core symbolic modules are proprietary to SpiralReality.
