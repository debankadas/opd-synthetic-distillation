#!/usr/bin/env python3
"""
Generate domain-specific synthetic data for custom use cases.

This script allows you to generate training data tailored to specific domains,
industries, or company use cases.

Usage:
    # Generate healthcare domain data
    python examples/generate_custom_domain.py --domain healthcare --num-records 500
    
    # Generate finance domain data with custom config
    python examples/generate_custom_domain.py --domain finance --config custom_domains.json
    
    # Generate from custom YAML config
    python examples/generate_custom_domain.py --config-file my_domain.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infra.synthetic_data_generator import (
    SyntheticDataGenerator,
    save_as_conversations,
)


# Predefined domain configurations
DOMAIN_CONFIGS = {
    "healthcare": {
        "reasoning_tasks": {
            "medical_diagnosis": 3.0,
            "treatment_planning": 2.0,
            "clinical_reasoning": 2.0,
            "patient_case_analysis": 1.5,
        },
        "dialogues": [
            "doctor_patient",
            "nurse_patient",
            "medical_consultation",
            "health_advice",
        ],
        "instructions": {
            "medical_writing": ["case_reports", "treatment_plans", "patient_education"],
            "clinical_tasks": ["diagnosis", "prescription", "lab_interpretation"],
            "healthcare_admin": ["scheduling", "insurance", "medical_records"],
        },
    },
    "finance": {
        "reasoning_tasks": {
            "financial_analysis": 3.0,
            "investment_reasoning": 2.5,
            "risk_assessment": 2.0,
            "market_analysis": 1.5,
        },
        "dialogues": [
            "financial_advisor",
            "banking_support",
            "investment_consultation",
            "financial_planning",
        ],
        "instructions": {
            "financial_analysis": ["portfolio_review", "risk_analysis", "market_research"],
            "reporting": ["financial_statements", "investment_reports", "summaries"],
            "planning": ["budgeting", "tax_planning", "retirement_planning"],
        },
    },
    "legal": {
        "reasoning_tasks": {
            "legal_reasoning": 3.0,
            "case_analysis": 2.5,
            "contract_review": 2.0,
            "regulatory_compliance": 1.5,
        },
        "dialogues": [
            "lawyer_client",
            "legal_consultation",
            "contract_negotiation",
            "compliance_advice",
        ],
        "instructions": {
            "legal_writing": ["contracts", "briefs", "legal_memos"],
            "analysis": ["case_review", "statute_interpretation", "precedent_research"],
            "compliance": ["regulatory_review", "policy_drafting", "due_diligence"],
        },
    },
    "customer_support": {
        "reasoning_tasks": {
            "troubleshooting": 3.0,
            "problem_solving": 2.5,
            "policy_interpretation": 1.5,
        },
        "dialogues": [
            "technical_support",
            "customer_service",
            "product_support",
            "complaint_resolution",
        ],
        "instructions": {
            "support_tasks": ["issue_resolution", "product_guidance", "refund_processing"],
            "documentation": ["knowledge_base", "faqs", "tutorials"],
            "communication": ["email_responses", "chat_scripts", "escalation_handling"],
        },
    },
    "education": {
        "reasoning_tasks": {
            "pedagogical_reasoning": 2.5,
            "curriculum_design": 2.0,
            "assessment_creation": 2.0,
            "learning_analytics": 1.5,
        },
        "dialogues": [
            "educational_tutoring",
            "student_guidance",
            "teacher_consultation",
            "academic_advising",
        ],
        "instructions": {
            "teaching": ["lesson_planning", "assessment_design", "feedback"],
            "content_creation": ["study_guides", "practice_problems", "explanations"],
            "administration": ["grading", "progress_tracking", "course_planning"],
        },
    },
    "ecommerce": {
        "reasoning_tasks": {
            "product_recommendation": 3.0,
            "pricing_strategy": 2.0,
            "inventory_optimization": 1.5,
        },
        "dialogues": [
            "product_inquiry",
            "purchase_assistance",
            "returns_support",
            "product_reviews",
        ],
        "instructions": {
            "product_management": ["descriptions", "categorization", "seo_optimization"],
            "customer_engagement": ["recommendations", "upselling", "loyalty_programs"],
            "operations": ["order_processing", "shipping", "returns_handling"],
        },
    },
}


def load_custom_config(config_path: str) -> Dict:
    """Load custom domain configuration from JSON or YAML."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    elif path.suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .json or .yaml")


def generate_domain_dataset(
    domain: str,
    num_records: int = 500,
    output_dir: str = "synthetic_datasets",
    backend: str = "openai",
    custom_config: Optional[Dict] = None,
) -> Path:
    """
    Generate domain-specific dataset.
    
    Args:
        domain: Domain name (e.g., 'healthcare', 'finance') or 'custom'
        num_records: Total number of examples
        output_dir: Output directory
        backend: 'openai' or 'nemo'
        custom_config: Custom domain configuration (overrides predefined)
    
    Returns:
        Path to generated JSON file
    """
    # Get domain configuration
    if custom_config:
        config = custom_config
    elif domain in DOMAIN_CONFIGS:
        config = DOMAIN_CONFIGS[domain]
    else:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Available: {list(DOMAIN_CONFIGS.keys())} or provide --config-file"
        )
    
    print(f"\n🎯 Generating {num_records} examples for domain: {domain}")
    print(f"   Backend: {backend}")
    
    # Initialize generator
    generator = SyntheticDataGenerator(backend=backend)
    
    # Split records across data types
    reasoning_ratio = 0.5
    dialogue_ratio = 0.3
    instruction_ratio = 0.2
    
    num_reasoning = int(num_records * reasoning_ratio)
    num_dialogue = int(num_records * dialogue_ratio)
    num_instruction = int(num_records * instruction_ratio)
    
    datasets = []
    
    # Generate reasoning data
    if "reasoning_tasks" in config and num_reasoning > 0:
        print(f"\n📊 Generating {num_reasoning} reasoning examples...")
        df_reasoning = generator.generate_reasoning_dataset(
            num_records=num_reasoning,
            task_weights=config["reasoning_tasks"],
        )
        datasets.append(df_reasoning)
    
    # Generate dialogue data
    if "dialogues" in config and num_dialogue > 0:
        print(f"\n💬 Generating {num_dialogue} dialogue examples...")
        df_dialogue = generator.generate_dialogue_dataset(
            num_records=num_dialogue,
            domains=config["dialogues"],
        )
        datasets.append(df_dialogue)
    
    # Generate instruction data
    if "instructions" in config and num_instruction > 0:
        print(f"\n📝 Generating {num_instruction} instruction examples...")
        df_instruction = generator.generate_instruction_dataset(
            num_records=num_instruction,
            categories=config["instructions"],
        )
        datasets.append(df_instruction)
    
    # Combine datasets
    import pandas as pd
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path / f"{domain}_{num_records}.csv"
    json_path = output_path / f"{domain}_{num_records}.json"
    
    combined_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved CSV: {csv_path}")
    
    # Convert to OPD format
    save_as_conversations(combined_df, str(json_path))
    print(f"✅ Saved OPD JSON: {json_path}")
    
    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate domain-specific synthetic training data"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        default="healthcare",
        help=f"Domain name. Available: {list(DOMAIN_CONFIGS.keys())}",
    )
    
    parser.add_argument(
        "--num-records",
        type=int,
        default=500,
        help="Number of examples to generate (default: 500)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="synthetic_datasets",
        help="Output directory (default: synthetic_datasets)",
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "nemo"],
        help="Generation backend (default: openai)",
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to custom domain config (JSON or YAML)",
    )
    
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List available predefined domains and exit",
    )
    
    args = parser.parse_args()
    
    # List domains and exit
    if args.list_domains:
        print("\n📋 Available Predefined Domains:\n")
        for domain, config in DOMAIN_CONFIGS.items():
            print(f"  • {domain}")
            if "reasoning_tasks" in config:
                print(f"    Reasoning tasks: {list(config['reasoning_tasks'].keys())[:3]}...")
            if "dialogues" in config:
                print(f"    Dialogue types: {config['dialogues'][:3]}...")
        print("\nUse --domain DOMAIN_NAME to generate data")
        print("Or use --config-file to provide custom configuration")
        return
    
    # Load custom config if provided
    custom_config = None
    if args.config_file:
        print(f"\n📄 Loading custom config: {args.config_file}")
        custom_config = load_custom_config(args.config_file)
        args.domain = "custom"
    
    # Generate dataset
    json_path = generate_domain_dataset(
        domain=args.domain,
        num_records=args.num_records,
        output_dir=args.output_dir,
        backend=args.backend,
        custom_config=custom_config,
    )
    
    print(f"\n✨ Domain-specific dataset ready for OPD training!")
    print(f"\nNext steps:")
    print(f"  1. Update .env: DATA_PATH={json_path}")
    print(f"  2. Run training: python -m src.app.train_gold")


if __name__ == "__main__":
    main()
