### What is Aether?
Aether is a **monolithic, PyTorch-first runflow** for reproducible experiments:
- Single-file orchestration + config-driven runs (auto-detects `aether.config.json`/`configs/*.json`)
- Built-in logging & trace replay
- Optimizer stack: AdamW (decoupled weight decay), cosine/OneCycle schedulers,
  AMP/bfloat16, gradient clipping, EMA
- MPS acceleration (Apple Silicon) with torch.compile and alignment padding

### Configuration
- Drop a JSON file named `aether.config.json` (or a single `.json` under `configs/`) to override CLI defaults automatically.
- Override precedence: CLI args > config file > defaults. Set `auto_mps_7b: false` or use `--no_auto_mps_7b` to opt out of automatic tuning.
- Even without a config file the runner will self-tune defaults: it scans `data/` or `datasets/` folders for corpora, enables `--train_stream`, aligns pack/buffer sizes, and picks a safe MPS token budget automatically.

> This is a **Public Spec build**. Core symbolic modules are proprietary to SpiralReality.
