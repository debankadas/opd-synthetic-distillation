#!/usr/bin/env python3
"""
Example script demonstrating Synthetic Data Generation + OPD integration.

This script:
1. Generates synthetic reasoning data using OpenAI (or optionally NeMo)
2. Saves it in OPD-compatible format
3. Shows how to validate data quality with the teacher model

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="sk-xxx"
    
    # Generate 1K reasoning examples
    python examples/synthetic_opd_demo.py --num-records 1000
    
    # Or use NeMo backend
    export NVIDIA_API_KEY="nvapi-xxx"
    python examples/synthetic_opd_demo.py --num-records 1000 --backend nemo
    
    # Then train with OPD
    export DATA_PATH=synthetic_datasets/reasoning_1000.json
    python -m src.app.train_gold
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file to get API keys
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infra.synthetic_data_generator import (
    SyntheticDataGenerator,
    GeneratorConfig,
    save_as_conversations,
)


def generate_comprehensive_dataset(
    num_records: int = 1000,
    output_dir: str = "synthetic_datasets",
    backend: str = "openai",
) -> Path:
    """
    Generate a comprehensive dataset for OPD training.
    
    Combines reasoning, dialogue, and instruction-following tasks
    with appropriate weighting.
    
    Args:
        num_records: Total number of examples to generate
        output_dir: Directory to save datasets
    
    Returns:
        Path to the combined JSON dataset
    """
    import pandas as pd
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # API key will be auto-detected based on backend
    if backend == "openai":
        api_key_env = "OPENAI_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} not set. Get one from: https://platform.openai.com/api-keys")
    else:  # nemo
        api_key_env = "NVIDIA_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} not set. Get one from: https://build.nvidia.com/explore/discover")
    
    generator = SyntheticDataGenerator(backend=backend, api_key=api_key)
    
    # Allocate records across task types
    # 50% reasoning, 30% instruction, 20% dialogue
    num_reasoning = int(num_records * 0.5)
    num_instruction = int(num_records * 0.3)
    num_dialogue = num_records - num_reasoning - num_instruction
    
    print(f"Generating {num_records} total examples:")
    print(f"  - {num_reasoning} reasoning tasks")
    print(f"  - {num_instruction} instruction-following")
    print(f"  - {num_dialogue} dialogue")
    print()
    
    # Generate each type
    print("Generating reasoning dataset...")
    df_reasoning = generator.generate_reasoning_dataset(
        num_records=num_reasoning,
        task_weights={
            "mathematical_reasoning": 2.0,
            "logical_reasoning": 2.0,
            "common_sense": 1.0,
            "reading_comprehension": 1.5,
        }
    )
    df_reasoning["dataset_type"] = "reasoning"
    df_reasoning.to_csv(output_path / "reasoning.csv", index=False)
    print(f"  Saved {len(df_reasoning)} reasoning examples")
    
    print("Generating instruction dataset...")
    df_instruction = generator.generate_instruction_dataset(
        num_records=num_instruction,
    )
    df_instruction["dataset_type"] = "instruction"
    df_instruction.to_csv(output_path / "instruction.csv", index=False)
    print(f"  Saved {len(df_instruction)} instruction examples")
    
    print("Generating dialogue dataset...")
    df_dialogue = generator.generate_dialogue_dataset(
        num_records=num_dialogue,
    )
    df_dialogue["dataset_type"] = "dialogue"
    df_dialogue.to_csv(output_path / "dialogue.csv", index=False)
    print(f"  Saved {len(df_dialogue)} dialogue examples")
    
    # Combine into single dataset
    print("\nCombining datasets...")
    
    # Normalize column names
    df_reasoning = df_reasoning.rename(columns={"question": "user", "answer": "assistant"})
    df_instruction = df_instruction.rename(columns={"instruction": "user", "response": "assistant"})
    df_dialogue = df_dialogue.rename(columns={"user_message": "user", "assistant_response": "assistant"})
    
    # Combine
    all_dfs = [df_reasoning, df_instruction, df_dialogue]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save combined CSV
    combined_csv = output_path / f"combined_{num_records}.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"Saved combined dataset to {combined_csv}")
    
    # Convert to OPD JSON format
    json_path = output_path / f"combined_{num_records}.json"
    save_as_conversations(
        combined_df,
        str(json_path),
        user_col="user",
        assistant_col="assistant",
        metadata_cols=["dataset_type", "task_type", "difficulty", "category", "domain"],
    )
    print(f"Saved OPD-compatible JSON to {json_path}")
    
    return json_path


def validate_with_teacher(
    json_path: Path,
    num_samples: int = 10,
    teacher_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
) -> None:
    """
    Validate generated data by scoring with teacher model.
    
    This checks if the teacher model agrees with NeMo-generated answers.
    High agreement suggests good data quality.
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\nValidating {num_samples} samples with teacher model...")
    print(f"Teacher: {teacher_model_id}")
    
    # Load data
    with open(json_path) as f:
        data = json.load(f)
    
    conversations = data.get("conversations", [])
    if num_samples > len(conversations):
        num_samples = len(conversations)
    
    # Sample random examples
    import random
    samples = random.sample(conversations, num_samples)
    
    # Load teacher (on CPU for quick validation)
    print("Loading teacher model...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    
    # Validate samples
    agreements = []
    for i, conv in enumerate(samples, 1):
        messages = conv["messages"]
        user_msg = messages[0]["content"]
        nemo_answer = messages[1]["content"]
        
        # Generate teacher's answer
        prompt = f"Question: {user_msg}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        teacher_answer = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        # Simple agreement check (first 50 chars)
        nemo_start = nemo_answer[:50].lower().strip()
        teacher_start = teacher_answer[:50].lower().strip()
        
        # Rough similarity (shared words)
        nemo_words = set(nemo_start.split())
        teacher_words = set(teacher_start.split())
        if len(nemo_words | teacher_words) > 0:
            overlap = len(nemo_words & teacher_words) / len(nemo_words | teacher_words)
        else:
            overlap = 0.0
        
        agreements.append(overlap)
        
        print(f"\n[{i}/{num_samples}]")
        print(f"Question: {user_msg[:100]}...")
        print(f"NeMo answer: {nemo_start}...")
        print(f"Teacher answer: {teacher_start}...")
        print(f"Overlap: {overlap:.2%}")
    
    avg_agreement = sum(agreements) / len(agreements)
    print(f"\n{'='*60}")
    print(f"Average overlap: {avg_agreement:.2%}")
    print(f"Quality assessment: ", end="")
    if avg_agreement > 0.6:
        print("✅ GOOD - High teacher agreement")
    elif avg_agreement > 0.4:
        print("⚠️  MODERATE - Acceptable quality")
    else:
        print("❌ LOW - Consider regenerating with different prompts")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate and validate NeMo dataset for OPD"
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=1000,
        help="Total number of examples to generate",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "nemo"],
        default="openai",
        help="Backend to use (default: openai)",
    )
    parser.add_argument(
        "--output-dir",
        default="synthetic_datasets",
        help="Directory to save datasets",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate with teacher model (slow)",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=10,
        help="Number of samples to validate",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation (only validate existing dataset)",
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    if not args.skip_generation:
        json_path = generate_comprehensive_dataset(
            num_records=args.num_records,
            output_dir=args.output_dir,
            backend=args.backend,
        )
    else:
        # Find existing dataset
        output_path = Path(args.output_dir)
        json_files = list(output_path.glob("combined_*.json"))
        if not json_files:
            print(f"No datasets found in {output_path}")
            sys.exit(1)
        json_path = json_files[-1]  # Use latest
        print(f"Using existing dataset: {json_path}")
    
    # Validate if requested
    if args.validate:
        validate_with_teacher(
            json_path,
            num_samples=args.validate_samples,
        )
    
    # Print next steps
    print("\n" + "="*60)
    print("✅ Dataset generation complete!")
    print("\nNext steps:")
    print(f"1. Review the data: {json_path}")
    print(f"2. Set environment variable:")
    print(f"   export DATA_PATH={json_path.absolute()}")
    print(f"3. Train with OPD:")
    print(f"   python -m src.app.train_gold")
    print("="*60)
