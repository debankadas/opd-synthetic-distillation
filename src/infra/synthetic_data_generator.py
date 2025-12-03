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
import random
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
    
    def _generate_with_openai(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
    ) -> str:
        """Helper to generate text with OpenAI API."""
        if model is None:
            model = self.config.openai_model
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1000,
        )
        return response.choices[0].message.content or ""
    
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
        if self.backend == "openai":
            return self._generate_reasoning_openai(num_records, task_weights)
        else:
            return self._generate_reasoning_nemo(num_records, task_weights)
    
    def _generate_reasoning_openai(
        self,
        num_records: int,
        task_weights: Optional[Dict[str, float]],
    ) -> pd.DataFrame:
        """Generate reasoning dataset using OpenAI."""
        if task_weights is None:
            task_weights = {
                "mathematical_reasoning": 2.0,
                "logical_reasoning": 2.0,
                "common_sense": 1.0,
                "reading_comprehension": 1.5,
            }
        
        task_types = list(task_weights.keys())
        weights = [task_weights[t] for t in task_types]
        difficulties = ["easy", "medium", "hard"]
        difficulty_weights = [1, 2, 1]
        
        records = []
        print(f"Generating {num_records} reasoning examples with OpenAI...")
        
        for i in range(num_records):
            # Sample task type and difficulty
            task_type = random.choices(task_types, weights=weights)[0]
            difficulty = random.choices(difficulties, weights=difficulty_weights)[0]
            
            # Generate question
            q_prompt = (
                f"Create a {difficulty} difficulty {task_type.replace('_', ' ')} question. "
                f"Make it clear, specific, and engaging. "
                f"Respond with only the question, no preamble."
            )
            q_system = (
                "You are an expert at creating educational questions. "
                "Generate diverse, high-quality questions that test understanding."
            )
            question = self._generate_with_openai(q_prompt, q_system, self.config.openai_model)
            
            # Generate answer
            a_prompt = (
                f"Answer this {task_type.replace('_', ' ')} question: {question}\n\n"
                f"Provide a clear, step-by-step explanation. Show your reasoning process."
            )
            a_system = (
                "You are a helpful assistant that provides detailed, accurate answers. "
                "Break down complex problems into steps. Be thorough but concise."
            )
            answer = self._generate_with_openai(a_prompt, a_system, self.config.openai_answer_model)
            
            records.append({
                "task_type": task_type,
                "difficulty": difficulty,
                "question": question.strip(),
                "answer": answer.strip(),
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_records} examples...")
        
        return pd.DataFrame(records)
    
    def _generate_reasoning_nemo(
        self,
        num_records: int,
        task_weights: Optional[Dict[str, float]],
    ) -> pd.DataFrame:
        """Generate reasoning dataset using NeMo."""
        if task_weights is None:
            task_weights = {
                "mathematical_reasoning": 2.0,
                "logical_reasoning": 2.0,
                "common_sense": 1.0,
                "reading_comprehension": 1.5,
            }
        
        config_builder = DataDesignerConfigBuilder()
        
        task_types = list(task_weights.keys())
        weights = [task_weights[t] for t in task_types]
        
        config_builder.add_column(
            SamplerColumnConfig(
                name="task_type",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(values=task_types, weights=weights),
            )
        )
        
        config_builder.add_column(
            SamplerColumnConfig(
                name="difficulty",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(
                    values=["easy", "medium", "hard"],
                    weights=[1, 2, 1],
                ),
            )
        )
        
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
                model_alias=self.config.nemo_model,
            )
        )
        
        config_builder.add_column(
            LLMTextColumnConfig(
                name="answer",
                prompt=(
                    "Answer this {{ task_type }} question: {{ question }}\\n\\n"
                    "Provide a clear, step-by-step explanation. "
                    "Show your reasoning process."
                ),
                system_prompt=(
                    "You are a helpful assistant that provides detailed, accurate answers. "
                    "Break down complex problems into steps. Be thorough but concise."
                ),
                model_alias=self.config.nemo_answer_model,
            )
        )
        
        preview = self.nemo_client.preview(config_builder, num_records=num_records)
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
            DataFrame with columns: domain, user_message, assistant_response
        """
        if self.backend == "openai":
            return self._generate_dialogue_openai(num_records, domains)
        else:
            return self._generate_dialogue_nemo(num_records, domains)
    
    def _generate_dialogue_openai(
        self,
        num_records: int,
        domains: Optional[List[str]],
    ) -> pd.DataFrame:
        """Generate dialogue dataset using OpenAI."""
        if domains is None:
            domains = [
                "customer_support",
                "technical_help",
                "general_chat",
                "educational_tutoring",
            ]
        
        records = []
        print(f"Generating {num_records} dialogue examples with OpenAI...")
        
        for i in range(num_records):
            domain = random.choice(domains)
            
            # Generate user message
            u_prompt = (
                f"Write a realistic {domain.replace('_', ' ')} question or request. "
                f"Make it specific and natural."
            )
            user_message = self._generate_with_openai(u_prompt, model=self.config.openai_model)
            
            # Generate assistant response
            a_prompt = f"Respond helpfully to this {domain.replace('_', ' ')} message: {user_message}"
            a_system = "You are a professional, helpful assistant. Provide accurate, friendly responses."
            assistant_response = self._generate_with_openai(a_prompt, a_system, self.config.openai_answer_model)
            
            records.append({
                "domain": domain,
                "user_message": user_message.strip(),
                "assistant_response": assistant_response.strip(),
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_records} examples...")
        
        return pd.DataFrame(records)
    
    def _generate_dialogue_nemo(
        self,
        num_records: int,
        domains: Optional[List[str]],
    ) -> pd.DataFrame:
        """Generate dialogue dataset using NeMo."""
        if domains is None:
            domains = ["customer_support", "technical_help", "general_chat", "educational_tutoring"]
        
        config_builder = DataDesignerConfigBuilder()
        
        config_builder.add_column(
            SamplerColumnConfig(
                name="domain",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(values=domains),
            )
        )
        
        config_builder.add_column(
            LLMTextColumnConfig(
                name="user_message",
                prompt="Write a {{ domain }} question or request. Make it realistic and specific.",
                model_alias=self.config.nemo_model,
            )
        )
        
        config_builder.add_column(
            LLMTextColumnConfig(
                name="assistant_response",
                prompt="You are a helpful assistant in {{ domain }}. Respond to: {{ user_message }}",
                system_prompt="You are a professional, helpful assistant. Provide accurate, friendly responses.",
                model_alias=self.config.nemo_answer_model,
            )
        )
        
        preview = self.nemo_client.preview(config_builder, num_records=num_records)
        return preview.dataset
    
    def generate_instruction_dataset(
        self,
        num_records: int = 100,
        categories: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Generate instruction-following dataset.
        
        Args:
            num_records: Number of instruction examples
            categories: Dict mapping categories to subcategories
        
        Returns:
            DataFrame with columns: category, subcategory, instruction, response
        """
        if self.backend == "openai":
            return self._generate_instruction_openai(num_records, categories)
        else:
            return self._generate_instruction_nemo(num_records, categories)
    
    def _generate_instruction_openai(
        self,
        num_records: int,
        categories: Optional[Dict[str, List[str]]],
    ) -> pd.DataFrame:
        """Generate instruction dataset using OpenAI."""
        if categories is None:
            categories = {
                "writing": ["email", "essay", "summary", "creative"],
                "coding": ["debug", "explain", "optimize", "implement"],
                "analysis": ["data", "text", "comparison", "evaluation"],
                "planning": ["project", "event", "travel", "schedule"],
            }
        
        records = []
        print(f"Generating {num_records} instruction examples with OpenAI...")
        
        for i in range(num_records):
            category = random.choice(list(categories.keys()))
            subcategory = random.choice(categories[category])
            
            # Generate instruction
            i_prompt = (
                f"Create a {category} task instruction specifically about {subcategory}. "
                f"Make it clear and actionable. Include relevant context."
            )
            instruction = self._generate_with_openai(i_prompt, model=self.config.openai_model)
            
            # Generate response
            r_prompt = (
                f"Complete this {category} task: {instruction}\n\n"
                f"Focus on {subcategory}. Provide a thorough, high-quality response."
            )
            r_system = "You are an expert at following instructions precisely. Provide complete, helpful responses."
            response = self._generate_with_openai(r_prompt, r_system, self.config.openai_answer_model)
            
            records.append({
                "category": category,
                "subcategory": subcategory,
                "instruction": instruction.strip(),
                "response": response.strip(),
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_records} examples...")
        
        return pd.DataFrame(records)
    
    def _generate_instruction_nemo(
        self,
        num_records: int,
        categories: Optional[Dict[str, List[str]]],
    ) -> pd.DataFrame:
        """Generate instruction dataset using NeMo."""
        if categories is None:
            categories = {
                "writing": ["email", "essay", "summary", "creative"],
                "coding": ["debug", "explain", "optimize", "implement"],
                "analysis": ["data", "text", "comparison", "evaluation"],
                "planning": ["project", "event", "travel", "schedule"],
            }
        
        config_builder = DataDesignerConfigBuilder()
        
        config_builder.add_column(
            SamplerColumnConfig(
                name="category",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(values=list(categories.keys())),
            )
        )
        
        config_builder.add_column(
            SamplerColumnConfig(
                name="subcategory",
                sampler_type=SamplerType.SUBCATEGORY,
                params=SubcategorySamplerParams(category="category", values=categories),
            )
        )
        
        config_builder.add_column(
            LLMTextColumnConfig(
                name="instruction",
                prompt="Create a {{ category }} task instruction about {{ subcategory }}. Make it clear and actionable.",
                system_prompt="You create clear, specific task instructions.",
                model_alias=self.config.nemo_model,
            )
        )
        
        config_builder.add_column(
            LLMTextColumnConfig(
                name="response",
                prompt="Complete this {{ category }} task: {{ instruction }}\\n\\nFocus on {{ subcategory }}.",
                system_prompt="You are an expert at following instructions precisely. Provide complete, helpful responses.",
                model_alias=self.config.nemo_answer_model,
            )
        )
        
        preview = self.nemo_client.preview(config_builder, num_records=num_records)
        return preview.dataset


def save_as_conversations(
    df: pd.DataFrame,
    output_path: str,
    user_col: str = "question",
    assistant_col: str = "answer",
    metadata_cols: Optional[List[str]] = None,
) -> None:
    """
    Save DataFrame as OPD-compatible conversation JSON.
    
    Args:
        df: Generated DataFrame
        output_path: Path to save JSON file
        user_col: Column name for user messages
        assistant_col: Column name for assistant messages
        metadata_cols: Additional columns to include as metadata
    """
    conversations = []
    for _, row in df.iterrows():
        conv = {
            "messages": [
                {"role": "user", "content": str(row[user_col])},
                {"role": "assistant", "content": str(row[assistant_col])},
            ]
        }
        
        if metadata_cols:
            conv["metadata"] = {col: row[col] for col in metadata_cols if col in row}
        
        conversations.append(conv)
    
    output_data = {"conversations": conversations}
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(conversations)} conversations to {output_path}")


def load_synthetic_dataset(csv_path: str) -> List[Dict]:
    """
    Load synthetic CSV output for use with OPD.
    
    Args:
        csv_path: Path to generated CSV
    
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
        raise ValueError(f"Could not find user/assistant columns. Found: {df.columns.tolist()}")
    
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
    
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--backend", choices=["openai", "nemo"], default="openai",
                        help="Backend to use (default: openai)")
    parser.add_argument("--type", choices=["reasoning", "dialogue", "instruction"], 
                        default="reasoning", help="Dataset type")
    parser.add_argument("--num-records", type=int, default=100, 
                        help="Number of examples to generate")
    parser.add_argument("--output", default="synthetic_data.csv", 
                        help="Output CSV path")
    parser.add_argument("--output-json", help="Also save as OPD-compatible JSON")
    parser.add_argument("--api-key", help="API key (or set OPENAI_API_KEY/NVIDIA_API_KEY)")
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(backend=args.backend, api_key=args.api_key)
    
    # Generate
    print(f"Generating {args.num_records} {args.type} examples using {args.backend}...")
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
        user_col = "question" if "question" in df.columns else (
            "instruction" if "instruction" in df.columns else "user_message"
        )
        assistant_col = "answer" if "answer" in df.columns else (
            "response" if "response" in df.columns else "assistant_response"
        )
        metadata_cols = [c for c in df.columns if c not in [user_col, assistant_col]]
        
        save_as_conversations(
            df, 
            args.output_json, 
            user_col=user_col,
            assistant_col=assistant_col,
            metadata_cols=metadata_cols,
        )
