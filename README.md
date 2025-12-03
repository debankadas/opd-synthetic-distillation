# On-Policy Distillation with Synthetic Data Generation

An end-to-end framework that pairs GOLD-style on-policy distillation with synthetic data generation so you can train smaller students from stronger teachers using minimal manual data.

## Quick Links

- examples/nemo_opd_demo.py - End-to-end synthetic data + OPD demo
- examples/generate_custom_domain.py - Domain-specific data generator
- docs/NEMO_INTEGRATION.md - Synthetic data integration (OpenAI + NeMo)
- scripts/run_eval.sh - Benchmark runner
- .env_example - Baseline configuration template

## What is OPD?

- The student generates rollouts; a teacher scores them; GOLD loss emphasizes tokens from high-score continuations to reduce distribution shift.
- SyntheticDataGenerator bootstraps mixed reasoning/dialogue/instruction data before training to widen coverage.
- Teacher backends are pluggable: OpenRouter API, llama.cpp GGUF, or full Hugging Face models.

## End-to-End Flow

1. Install dependencies and set API keys.
2. Generate a synthetic dataset.
3. Choose a teacher backend (OpenRouter, GGUF, or HF).
4. Train with on-policy distillation.
5. Evaluate against the baseline.
6. (Optional) Validate generated data quality.

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (or NVIDIA API key for NeMo backend)
- 8GB+ RAM (16GB+ recommended for training)
- pip/virtualenv and git

### 1. Set up environment

```bash
git clone <repo-url>
cd OPD
pip install -r requirements.txt
cp .env_example .env  # or: copy .env_example .env on Windows
```

Add your API key to `.env` or export it:

```bash
export OPENAI_API_KEY="sk-proj-your-key-here"
# or NeMo
export NVIDIA_API_KEY="nvapi-your-key-here"
```

### 2. Generate synthetic dataset

```bash
python examples/nemo_opd_demo.py \
  --num-records 100 \
  --backend openai \
  --output-dir synthetic_datasets
```

Creates `synthetic_datasets/combined_100.csv` and `combined_100.json` (OPD-ready). Costs ~$0.10, ~1-2 minutes with OpenAI defaults.

### 3. Choose teacher backend

**OpenRouter (lightweight, recommended)**

```bash
cat >> .env <<'EOF'
TEACHER_BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=qwen/qwen-2.5-7b-instruct
EOF
```

Good quality/cost balance; no local downloads.

**GGUF via llama.cpp (local, efficient)**

```bash
cat >> .env <<'EOF'
TEACHER_BACKEND=gguf
TEACHER_GGUF=bartowski/Qwen2.5-7B-Instruct-GGUF
TEACHER_PREFERRED_QUANT=Q4_K_M
TEACHER_N_GPU_LAYERS=-1
EOF
```

Keeps inference local; installs `llama-cpp-python`.

**Hugging Face full model (local, highest quality)**

```bash
cat >> .env <<'EOF'
TEACHER_BACKEND=hf
TEACHER_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
TEACHER_DTYPE=bf16
EOF
```

Best quality; requires larger download and GPU/RAM.

### 4. Train with OPD

```bash
cat >> .env <<'EOF'
DATA_PATH=synthetic_datasets/combined_100.json
OUTPUT_DIR=outputs/synthetic_opd_100
MAX_STEPS=200
SAVE_EVERY=50
LEARNING_RATE=5e-5
MAX_NEW_TOKENS=128
EOF

python -m src.app.train_gold
```

The loop: student generates -> teacher scores -> GOLD loss updates the student -> checkpoints go to `OUTPUT_DIR`.

### 5. Evaluate against baseline

```bash
export KD_MODEL_PATH=outputs/synthetic_opd_100/best
export BASE_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
bash scripts/run_eval.sh
```

Reports and plots are written to `reports/`.

### 6. Validate data quality (optional)

```bash
python examples/nemo_opd_demo.py \
  --skip-generation \
  --validate \
  --validate-samples 10
```

Samples generated data and scores it with the teacher; aim for >60% overlap.

## Why OPD (vs SFT/KD/RLHF)

- On-policy vs offline KD: student trains on its own rollouts scored by the teacher, reducing distribution shift compared to static datasets.
- Cheaper and simpler than RLHF: uses a fixed teacher instead of a learned reward model; fewer finicky reward-shaping hyperparameters.
- Sample efficiency: GOLD weighting focuses updates on high-scoring tokens/trajectories, improving data efficiency over plain SFT.
- Synthetic bootstrap: `examples/nemo_opd_demo.py` + `SyntheticDataGenerator` produce mixed reasoning/dialogue/instruction data before training.
- Flexible teacher backends: start with OpenRouter for convenience; switch to GGUF/HF for offline or cost-sensitive runs.
- Expected gains: target +5-10% over the base Qwen2.5-0.5B on ARC/HellaSwag/TruthfulQA; plug in your measured numbers below.

## Inspiration and Future Enhancements

This project is inspired in part by Hugging Face Unlocking On-Policy Distillation for Any Model Family (https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation).

How we differ today:

- Alignment: heuristic char-span alignment and greedy token-string recovery; the HF GOLD recipe merges tokens and vocab more robustly across tokenizers.
- Vocab mapping: only exact token-string matches are mapped into the student vocab; unmatched tokens are dropped and the ULD fallback is stubbed.
- Training mix: fully on-policy; no lambda blend of offline + online data.
- Benchmarks/baselines: no published Countdown/GRPO baselines; HF reports GOLD vs ULD vs GRPO with tokenizer-similarity analysis.

Future Work:

- Implement the HF GOLD recipe for more robust token alignment and vocab mapping.
- Add support for the GRPO baseline and Countdown-style benchmarking.

Future Planned enhancements:

- Ingest teacher tokenizers to build 1:1 token mappings and a real ULD fallback for unmatched vocab.
- Implement token-merge sequence alignment (logprob merge) to handle tokenizer length mismatches.
- Add lambda-controlled offline/online blending and expose it in env/config.
- Ship a GRPO baseline script and a reproducible Countdown-style benchmark harness with tokenizer-similarity metrics.
- Optional TRL integration for users who want HF Trainer-based runs alongside this lightweight loop.

## Results

Replace with your runs from `reports/` (generated by `scripts/run_eval.sh`):

![alt text](images/1762068337269.png)

![alt text](images/1762068365240.png)

![alt text](images/1762068375145.png)

![alt text](images/1762068375145-2.png)

![alt text](images/1762068390679.png)

![alt text](images/1762068398272.png)

![alt text](images/1762068403670.png)

![alt text](images/1762068424432.png)

![alt text](images/1762068432600.png)

## Expected Costs and Times (OpenAI)

| Examples | Cost    | Generation Time | Training Time\* |
| -------- | ------- | --------------- | --------------- |
| 100      | ~$0.10  | 1-2 min         | 10-20 min       |
| 1,000    | ~$1.00  | 10-15 min       | 1-2 hrs         |
| 10,000   | ~$10.00 | 1.5-2 hrs       | 10-20 hrs       |

\*Training time depends on hardware (GPU/MPS/CPU).

## Project Structure

```
src/
  app/
    train_gold.py            # Main training loop
    run_distill.py           # Single-step distillation
    prepare_dataset.py       # Dataset loading
  domain/                    # Alignment and loss logic
  infra/
    synthetic_data_generator.py
    teacher_local_hf.py
    teacher_llamacpp.py
    teacher_openrouter.py
    student_model.py
examples/
  nemo_opd_demo.py
  generate_custom_domain.py
docs/
  NEMO_INTEGRATION.md
scripts/
  run_eval.sh
  train_gpu_hf.sh
  train_mac_local.sh
```

## Troubleshooting

- Missing API key: ensure `OPENAI_API_KEY` or `NVIDIA_API_KEY` is in `.env` or exported.
- Slow generation: check connectivity and rate limits; reduce batch size.
- CUDA OOM: lower `MAX_NEW_TOKENS`, enable LoRA (`KD_LORA=1`), or pick a smaller student.
- Low teacher overlap (<40%): improve prompts, use a stronger teacher (e.g., `gpt-4o`), or validate before training.

## Resources

- OpenAI API docs: https://platform.openai.com/docs
- NeMo Data Designer: https://docs.nvidia.com/nemo-microservices/
- GOLD paper: https://arxiv.org/abs/2306.13649

## Citation

```
@article{gold2023,
  title={On-Policy Distillation of Language Models},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2023}
}
```

## License

See LICENSE for details.

## Support

- Check logs in `runs/`
- Validate API keys
- Test with the 100-sample demo
- Manually spot-check generated data
