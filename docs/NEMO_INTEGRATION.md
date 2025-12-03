# Synthetic Data Generation + OPD Integration Quick Start

## Overview

This guide shows how to generate high-quality synthetic training data at scale using **OpenAI API** and integrate it with On-Policy Distillation.

> **Note**: This guide uses OpenAI by default. If you prefer NVIDIA NeMo Data Designer, see the [NeMo Backend section](#using-nemo-backend-optional).

## Prerequisites

1. **Get OpenAI API Key**
   - Visit: https://platform.openai.com/api-keys
   - Sign up / log in
   - Create a new API key
   - Estimated cost: ~$1-5 for 1000 examples with gpt-4o-mini

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # This includes: openai>=1.0.0
   ```

3. **Set API Key**
   ```bash
   export OPENAI_API_KEY="sk-xxx"
   ```

## Quick Start: Generate & Train

### Step 1: Generate Synthetic Dataset

Generate 1,000 diverse examples across reasoning, instruction, and dialogue tasks:

```bash
python examples/nemo_opd_demo.py --num-records 1000 --backend openai
```

This creates:
- `synthetic_datasets/combined_1000.csv` - Raw output
- `synthetic_datasets/combined_1000.json` - OPD-compatible format

### Step 2: Train with OPD

Update your `.env` file:
```bash
DATA_PATH=synthetic_datasets/combined_1000.json
OUTPUT_DIR=outputs/synthetic_opd_run
MAX_STEPS=500
SAVE_EVERY=100
```

Then train:
```bash
python -m src.app.train_gold
```

### Step 3: Evaluate

Compare your NeMo+OPD model against baseline:

```bash
# Set model paths in scripts/run_eval.sh
export KD_MODEL_PATH=outputs/synthetic_opd_run/best
export BASE_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct

bash scripts/run_eval.sh
```

## Advanced Usage

### Custom Dataset Types

Generate specific dataset types:

```python
from src.infra.synthetic_data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(backend="openai")

# Only reasoning tasks
df = generator.generate_reasoning_dataset(
    num_records=500,
    task_weights={
        "mathematical_reasoning": 3.0,
        "logical_reasoning": 2.0,
        "common_sense": 1.0,
    }
)
df.to_csv("reasoning_only.csv")

# Only dialogue
df = generator.generate_dialogue_dataset(
    num_records=500,
    domains=["customer_support", "technical_help"]
)
df.to_csv("dialogue_only.csv")

# Only instructions
df = generator.generate_instruction_dataset(num_records=500)
df.to_csv("instructions_only.csv")
```

### Validate Data Quality

Check teacher agreement on generated data:

```bash
python examples/nemo_opd_demo.py \
  --skip-generation \
  --validate \
  --validate-samples 20
```

This scores NeMo-generated answers using your teacher model. High overlap (>60%) indicates good quality.

### Iterative Refinement

1. **Train initial model**
   ```bash
   python -m src.app.train_gold
   ```

2. **Analyze weak areas**
   - Check training logs in `runs/`
   - Look for low agreement on specific categories
   - Example: "mathematical_reasoning" has low scores

3. **Generate targeted data**
   ```python
   # In Python
   from src.infra.nemo_data_generator import NemoDataGenerator
   
   generator = NemoDataGenerator()
   df = generator.generate_reasoning_dataset(
       num_records=1000,
       task_weights={
           "mathematical_reasoning": 10.0,  # Heavy focus
           "logical_reasoning": 1.0,
       }
   )
   ```

4. **Retrain with combined data**
   - Merge new data with previous dataset
   - Train for additional steps

## Integration Options

### Option 1: Pre-Generation (Current Implementation)

- Generate full dataset with NeMo **before** training
- Simple, decoupled, easy to debug
- **Best for**: Initial experiments, datasets <100K

### Option 2: Dynamic Generation (Future)

- Generate new examples **during** training based on weaknesses
- Adaptive curriculum learning
- **Best for**: Large-scale training, targeting specific skills

### Option 3: Teacher-Filtered (Future)

- Generate candidates with NeMo, filter with teacher scoring
- Quality-gated dataset
- **Best for**: High-quality, curated datasets

## Configuration Reference

### NeMo Model Aliases

Available in the free trial:

- `nemotron-nano-v2` - Fast, general-purpose (9B params)
- `nemotron-super` - High quality (49B params)
- `mistral-small` - Alternative strong model (24B)
- `llama-4-scout-17b` - Efficient mid-size model

**Recommendation**: Use `nemotron-nano-v2` for questions, `nemotron-super` for answers.

### Dataset Size Guidelines

| Use Case | Examples | Generation Time* | Cost** |
|----------|----------|------------------|--------|
| Pilot | 100 | 5-10 min | Free |
| Small | 1,000 | 30-60 min | Free |
| Medium | 10,000 | 5-10 hours | Pay*** |
| Large | 100,000 | 2-3 days | Pay*** |

*Approximate, depends on API speed  
**Free trial has limits  
***Requires paid plan after free tier

## Expected Results

### Baseline (No NeMo)
- Model: Qwen2.5-0.5B + OPD
- Data: 1K human-curated examples
- ARC-Easy: ~55%
- HellaSwag: ~40%

### With NeMo Integration
- Model: Qwen2.5-0.5B + OPD + NeMo data
- Data: 50K synthetic + 1K human
- **Expected**: +5-10pp on benchmarks
- **Goal**: Beat standalone Qwen2.5-0.5B baseline

## Troubleshooting

### API Key Issues
```
Error: Authentication failed
```
**Solution**: Ensure `OPENAI_API_KEY` is set correctly:
```bash
echo $OPENAI_API_KEY  # Should print your key (sk-...)
```

For NeMo backend:
```bash
echo $NVIDIA_API_KEY  # Should print your keyfor NeMo (nvapi-...)
```

### Import Errors
```
ModuleNotFoundError: No module named 'openai'
```
**Solution**: Install the package:
```bash
pip install openai
```

### Low Quality Data
- Teacher overlap < 40%
- Generated answers are generic/incorrect

**Solution**:
1. Improve prompts in `synthetic_data_generator.py`
2. Use stronger model (`gpt-4o` instead of `gpt-4o-mini`)
3. Add validation rules (length, keywords, etc.)
4. Try NeMo backend for different results

### Slow Generation
- Taking hours for 1K examples

**Solution**:
1. OpenAI is typically fast (~5-10 min for 1K)
2. Use `gpt-4o-mini` for speed (default)
3. Check API rate limits
4. Generate in smaller batches

## Using NeMo Backend (Optional)

If you prefer NVIDIA NeMo Data Designer instead of OpenAI:

1. **Get NVIDIA API Key**
   - Visit: https://build.nvidia.com/explore/discover
   - Sign up for free trial
   - Generate API key

2. **Install NeMo Package**
   ```bash
   pip install "nemo-microservices[data-designer]"
   ```

3. **Set API Key**
   ```bash
   export NVIDIA_API_KEY="nvapi-xxx"
   ```

4. **Generate with NeMo Backend**
   ```bash
   python examples/nemo_opd_demo.py --num-records 1000 --backend nemo
   ```

**NeMo vs OpenAI:**
- **OpenAI**: Faster, pay-per-use, familiar GPT models
- **NeMo**: Free trial, optimized for data generation, compound AI system

## Best Practices

1. **Start Small**: Generate 100-1K examples first, validate quality
2. **Mix Data**: Combine synthetic + human-curated for diversity
3. **Iterate**: Use training metrics to guide next generation round
4. **Validate**: Always check teacher agreement on samples
5. **Track Metadata**: Keep category/difficulty info for analysis

## Next Steps

1. **Baseline Evaluation**: Test Qwen2.5-0.5B on your target benchmarks
2. **Pilot Generation**: Create 1K synthetic examples
3. **First Training Run**: OPD with synthetic data
4. **Compare Results**: Measure improvement vs baseline
5. **Iterate**: If successful, scale to 10K-50K examples

## Resources

- [OpenAI API Docs](https://platform.openai.com/docs)
- [NeMo Data Designer Docs](https://docs.nvidia.com/nemo-microservices/) (optional)
- [Implementation Plan](../brain/.../implementation_plan.md)
- [OPD Documentation](docs/README.md)

## Support

If you encounter issues:
1. Check logs in `runs/` for detailed errors
2. Validate API key is active (OpenAI or NVIDIA)
3. Test with minimal example (100 records)
4. Review generated data manually for quality
