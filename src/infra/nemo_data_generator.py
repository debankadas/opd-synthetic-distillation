"""
Synthetic Data Generator for On-Policy Distillation.

Supports multiple backends:
- OpenAI API (default) - Uses GPT models for data generation
- NVIDIA NeMo Data Designer (optional) - Advanced synthetic data generation

Example usage:
    # Using OpenAI (default)
    generator = SyntheticDataGenerator(backend="openai", api_key="sk-...")
    dataset = generator.generate_reasoning_dataset(num_records=1000)
    dataset.to_csv("synthetic_reasoning_data.csv")
    
    # Using NeMo Data Designer
    generator = SyntheticDataGenerator(backend="nemo", api_key="nvapi-...")
    dataset = generator.generate_reasoning_dataset(num_records=1000)
    
    # Load into OPD
    conversations = load_synthetic_dataset("synthetic_reasoning_data.csv")
    # Use with train_gold.py via DATA_PATH
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Literal
from pathlib import Path
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from nemo_microservices.data_designer.essentials import (
        CategorySamplerParams,
        DataDesignerConfigBuilder,
        LLMTextColumnConfig,
        NeMoDataDesignerClient,
        SamplerColumnConfig,
        SamplerType,
        SubcategorySamplerParams,
        UniformSamplerParams,
    )
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

import pandas as pd


BackendType = Literal["openai", "nemo"]


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation."""
    backend: BackendType = "openai"
    api_key: Optional[str] = None
    # OpenAI specific
    openai_model: str = "gpt-4o-mini"  # Fast, cost-effective
    openai_answer_model: str = "gpt-4o"  # Better quality for answers
    # NeMo specific
    nemo_base_url: str = "https://ai.api.nvidia.com/v1/nemo/dd"
    nemo_model: str = "nemotron-nano-v2"
    nemo_answer_model: str = "nemotron-super"


class SyntheticDataGenerator:
    """Generate synthetic training data using OpenAI or NeMo."""
    
    def __init__(
        self,
        backend: BackendType = "openai",
        api_key: Optional[str] = None,
        config: Optional[GeneratorConfig] = None,
    ):
        if config is None:
            # Auto-detect API key
            if api_key is None:
                if backend == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("Set OPENAI_API_KEY or provide api_key")
                else:  # nemo
                    api_key = os.getenv("NVIDIA_API_KEY")
                    if not api_key:
                        raise ValueError("Set NVIDIA_API_KEY or provide api_key")
            
            config = GeneratorConfig(backend=backend, api_key=api_key)
        
        self.config = config
        self.backend = config.backend
        
        # Initialize backend
        if self.backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed. Install with: pip install openai")
            self.openai_client = OpenAI(api_key=config.api_key)
        elif self.backend == "nemo":
            if not NEMO_AVAILABLE:
                raise ImportError(
                    "nemo-microservices not installed. "
                    "Install with: pip install 'nemo-microservices[data-designer]'"
                )
            self.nemo_client = NeMoDataDesignerClient(
                base_url=config.nemo_base_url,
                default_headers={"Authorization": f"Bearer {config.api_key}"}
            )
    
    def generate_reasoning_dataset(
        self,
        num_records: int = 100,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Generate a dataset focused on reasoning tasks.
        
        Args:
            num_records: Number of examples to generate
            task_weights: Custom weights for task types (defaults to balanced)
        
        Returns:
            DataFrame with columns: task_type, difficulty, question, answer
        """
        config_builder = DataDesignerConfigBuilder()
        
        # Task type distribution
        if task_weights is None:
            task_weights = {
                "mathematical_reasoning": 2.0,
                "logical_reasoning": 2.0,
                "common_sense": 1.0,
                "reading_comprehension": 1.5,
            }
        
        task_types = list(task_weights.keys())
        weights = [task_weights[t] for t in task_types]
        
        config_builder.add_column(
            SamplerColumnConfig(
                name="task_type",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(
                    values=task_types,
                    weights=weights,
                ),
            )
        )
        
        # Difficulty levels
        config_builder.add_column(
            SamplerColumnConfig(
                name="difficulty",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(
                    values=["easy", "medium", "hard"],
                    weights=[1, 2, 1],  # Focus on medium
                ),
            )
        )
        
        # Generate question
        config_builder.add_column(
            LLMTextColumnConfig(
                name="question",
                prompt=(
                    "Create a {{ difficulty }} difficulty {{ task_type }} question. "
                    "Make it clear, specific, and engaging. "
                    "Respond with only the question, no preamble."
                ),
                system_prompt=(
                    "You are an expert at creating educational questions. "
                    "Generate diverse, high-quality questions that test understanding."
                ),
                model_alias=self.config.default_model,
            )
        )
        
        # Generate detailed answer
        config_builder.add_column(
            LLMTextColumnConfig(
                name="answer",
                prompt=(
                    "Answer this {{ task_type }} question: {{ question }}\n\n"
                    "Provide a clear, step-by-step explanation. "
                    "Show your reasoning process."
                ),
                system_prompt=(
                    "You are a helpful assistant that provides detailed, accurate answers. "
                    "Break down complex problems into steps. Be thorough but concise."
                ),
                model_alias=self.config.answer_model,
            )
        )
        
        # Generate the dataset
        preview = self.client.preview(config_builder, num_records=num_records)
        return preview.dataset
    
    def generate_dialogue_dataset(
        self,
        num_records: int = 100,
        domains: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate conversational dialogue dataset.
        
        Args:
            num_records: Number of conversations to generate
            domains: List of conversation domains
        
        Returns:
            DataFrame with columns: domain, persona, user_message, assistant_response
        """
        if domains is None:
            domains = [
                "customer_support",
                "technical_help",
                "general_chat",
                "educational_tutoring",
            ]
        
        config_builder = DataDesignerConfigBuilder()
        
        # Domain selection
        config_builder.add_column(
            SamplerColumnConfig(
                name="domain",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(values=domains),
            )
        )
        
        # User persona
        config_builder.add_column(
            LLMTextColumnConfig(
                name="persona",
                prompt=(
                    "Describe a user persona for a {{ domain }} conversation. "
                    "Include their background and what they might ask about. "
                    "Keep it brief (2-3 sentences)."
                ),
                model_alias=self.config.default_model,
            )
        )
        
        # User message
        config_builder.add_column(
            LLMTextColumnConfig(
                name="user_message",
                prompt=(
                    "Write a {{ domain }} question or request from this user: {{ persona }}. "
                    "Make it realistic and specific."
                ),
                model_alias=self.config.default_model,
            )
        )
        
        # Assistant response
        config_builder.add_column(
            LLMTextColumnConfig(
                name="assistant_response",
                prompt=(
                    "You are a helpful assistant in {{ domain }}. "
                    "Respond to this user message: {{ user_message }}\n\n"
                    "User context: {{ persona }}"
                ),
                system_prompt=(
                    "You are a professional, helpful assistant. "
                    "Provide accurate, friendly responses."
                ),
                model_alias=self.config.answer_model,
            )
        )
        
        preview = self.client.preview(config_builder, num_records=num_records)
        return preview.dataset
    
    def generate_instruction_dataset(
        self,
        num_records: int = 100,
        categories: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Generate instruction-following dataset with hierarchical categories.
        
        Args:
            num_records: Number of instruction examples
            categories: Dict mapping categories to subcategories
        
        Returns:
            DataFrame with columns: category, subcategory, instruction, response
        """
        if categories is None:
            categories = {
                "writing": ["email", "essay", "summary", "creative"],
                "coding": ["debug", "explain", "optimize", "implement"],
                "analysis": ["data", "text", "comparison", "evaluation"],
                "planning": ["project", "event", "travel", "schedule"],
            }
        
        config_builder = DataDesignerConfigBuilder()
        
        # Category
        config_builder.add_column(
            SamplerColumnConfig(
                name="category",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(values=list(categories.keys())),
            )
        )
        
        # Subcategory
        config_builder.add_column(
            SamplerColumnConfig(
                name="subcategory",
                sampler_type=SamplerType.SUBCATEGORY,
                params=SubcategorySamplerParams(
                    category="category",
                    values=categories,
                ),
            )
        )
        
        # Instruction
        config_builder.add_column(
            LLMTextColumnConfig(
                name="instruction",
                prompt=(
                    "Create a {{ category }} task instruction specifically about {{ subcategory }}. "
                    "Make it clear and actionable. Include relevant context."
                ),
                system_prompt="You create clear, specific task instructions.",
                model_alias=self.config.default_model,
            )
        )
        
        # Response
        config_builder.add_column(
            LLMTextColumnConfig(
                name="response",
                prompt=(
                    "Complete this {{ category }} task: {{ instruction }}\n\n"
                    "Focus on {{ subcategory }}. Provide a thorough, high-quality response."
                ),
                system_prompt=(
                    "You are an expert at following instructions precisely. "
                    "Provide complete, helpful responses."
                ),
                model_alias=self.config.answer_model,
            )
        )
        
        preview = self.client.preview(config_builder, num_records=num_records)
        return preview.dataset


def save_nemo_as_conversations(
    df: pd.DataFrame,
    output_path: str,
    user_col: str = "question",
    assistant_col: str = "answer",
    metadata_cols: Optional[List[str]] = None,
) -> None:
    """
    Save NeMo DataFrame as OPD-compatible conversation JSON.
    
    Args:
        df: NeMo output DataFrame
        output_path: Path to save JSON file
        user_col: Column name for user messages
        assistant_col: Column name for assistant messages
        metadata_cols: Additional columns to include as metadata
    """
    import json
    
    conversations = []
    for _, row in df.iterrows():
        conv = {
            "messages": [
                {"role": "user", "content": str(row[user_col])},
                {"role": "assistant", "content": str(row[assistant_col])},
            ]
        }
        
        # Add metadata
        if metadata_cols:
            conv["metadata"] = {col: row[col] for col in metadata_cols if col in row}
        
        conversations.append(conv)
    
    # Save in format compatible with prepare_dataset.py
    output_data = {"conversations": conversations}
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(conversations)} conversations to {output_path}")


def load_nemo_dataset(csv_path: str) -> List[Dict]:
    """
    Load NeMo CSV output for use with OPD.
    
    Args:
        csv_path: Path to NeMo-generated CSV
    
    Returns:
        List of conversation dicts (compatible with prepare_dataset.py)
    """
    df = pd.read_csv(csv_path)
    
    # Auto-detect columns
    possible_user_cols = ["question", "instruction", "user_message", "user", "prompt"]
    possible_assistant_cols = ["answer", "response", "assistant_response", "assistant", "completion"]
    
    user_col = None
    assistant_col = None
    
    for col in possible_user_cols:
        if col in df.columns:
            user_col = col
            break
    
    for col in possible_assistant_cols:
        if col in df.columns:
            assistant_col = col
            break
    
    if not user_col or not assistant_col:
        raise ValueError(
            f"Could not find user/assistant columns. "
            f"Found: {df.columns.tolist()}"
        )
    
    conversations = []
    for _, row in df.iterrows():
        conversations.append({
            "messages": [
                {"role": "user", "content": str(row[user_col])},
                {"role": "assistant", "content": str(row[assistant_col])},
            ]
        })
    
    return conversations


# CLI for quick generation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic data with NeMo")
    parser.add_argument("--type", choices=["reasoning", "dialogue", "instruction"], 
                        default="reasoning", help="Dataset type")
    parser.add_argument("--num-records", type=int, default=100, 
                        help="Number of examples to generate")
    parser.add_argument("--output", default="nemo_data.csv", 
                        help="Output CSV path")
    parser.add_argument("--output-json", help="Also save as OPD-compatible JSON")
    parser.add_argument("--api-key", help="NVIDIA API key (or set NVIDIA_API_KEY)")
    
    args = parser.parse_args()
    
    # Setup
    if args.api_key:
        config = NemoConfig(api_key=args.api_key)
    else:
        config = None  # Will use env var
    
    generator = NemoDataGenerator(config)
    
    # Generate
    print(f"Generating {args.num_records} {args.type} examples...")
    if args.type == "reasoning":
        df = generator.generate_reasoning_dataset(num_records=args.num_records)
    elif args.type == "dialogue":
        df = generator.generate_dialogue_dataset(num_records=args.num_records)
    else:  # instruction
        df = generator.generate_instruction_dataset(num_records=args.num_records)
    
    # Save CSV
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} records to {args.output}")
    
    # Save JSON if requested
    if args.output_json:
        # Detect columns
        user_col = "question" if "question" in df.columns else (
            "instruction" if "instruction" in df.columns else "user_message"
        )
        assistant_col = "answer" if "answer" in df.columns else (
            "response" if "response" in df.columns else "assistant_response"
        )
        metadata_cols = [c for c in df.columns if c not in [user_col, assistant_col]]
        
        save_nemo_as_conversations(
            df, 
            args.output_json, 
            user_col=user_col,
            assistant_col=assistant_col,
            metadata_cols=metadata_cols,
        )
