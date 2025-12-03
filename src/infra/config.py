from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DistillConfig:
    data_path: str
    student_model: str  # e.g., "mock" or "local:/path_or_hf_id"
    teacher_model: str  # e.g., "openrouter:qwen/qwen2.5-7b-instruct" or "mock"
    use_mock_teacher: bool
    num_samples: int
    beta: float = 0.0  # forward KL (teacher || pi) weight
    w_gkd: float = 1.0
    w_uld: float = 0.0
    max_new_tokens: int = 128
    top_k_student: int = 50
    device: str = "auto"  # auto|cpu|cuda

    @staticmethod
    def from_env_defaults() -> "DistillConfig":
        return DistillConfig(
            data_path=os.getenv(
                "DATA_PATH",
                "sample_training_data/retailco_training_data_sft_conversations.json",
            ),
            student_model=os.getenv("STUDENT_MODEL", "mock"),
            teacher_model=os.getenv(
                "TEACHER_MODEL", "openrouter:qwen/qwen2.5-7b-instruct"
            ),
            use_mock_teacher=os.getenv("USE_MOCK_TEACHER", "1") == "1",
            num_samples=int(os.getenv("NUM_SAMPLES", "4")),
        )
