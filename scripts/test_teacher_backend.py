#!/usr/bin/env python3
"""
Test script to verify teacher backend configuration.
Usage: python scripts/test_teacher_backend.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_teacher_backend():
    load_dotenv()
    
    backend = os.getenv("TEACHER_BACKEND", "hf")
    print(f"🔍 Testing Teacher Backend: {backend}")
    print("=" * 60)
    
    if backend == "openrouter":
        print("\n📡 OpenRouter Configuration:")
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")
        
        if not api_key or api_key == "":
            print("❌ OPENROUTER_API_KEY not set - will use MOCK mode")
            print("   To use real API, set OPENROUTER_API_KEY in .env")
        else:
            print(f"✅ API Key found: {api_key[:20]}...")
        
        print(f"✅ Model: {model}")
        
        # Test import
        try:
            from src.infra.teacher_openrouter import TeacherProvider
            model_spec = f"openrouter:{model}"
            use_mock = not api_key or api_key == ""
            teacher = TeacherProvider(model_spec=model_spec, use_mock=use_mock)
            
            # Test scoring
            test_prompt = "What is 2+2?"
            test_completion = "4"
            scores = teacher.score_completion_logprobs(test_prompt, test_completion)
            
            print(f"✅ Teacher initialized successfully")
            print(f"✅ Test scoring returned {len(scores.per_step)} steps")
            if use_mock:
                print("⚠️  Running in MOCK mode (for testing only)")
        except Exception as e:
            print(f"❌ Error initializing teacher: {e}")
            return False
            
    elif backend == "gguf":
        print("\n💾 GGUF Configuration:")
        gguf_path = os.getenv("TEACHER_GGUF_PATH")
        gguf_spec = os.getenv("TEACHER_GGUF")
        
        if gguf_path:
            path = Path(gguf_path)
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"✅ Local GGUF file found: {gguf_path}")
                print(f"   Size: {size_mb:.1f} MB")
            else:
                print(f"❌ TEACHER_GGUF_PATH set but file not found: {gguf_path}")
                if gguf_spec:
                    print(f"   Will download from: {gguf_spec}")
                return False
        elif gguf_spec:
            print(f"⚠️  TEACHER_GGUF_PATH not set, will download from: {gguf_spec}")
        else:
            print("❌ Neither TEACHER_GGUF_PATH nor TEACHER_GGUF is set")
            return False
        
        # Test import
        try:
            from src.infra.teacher_llamacpp import LlamaCppTeacher
            print("✅ llama-cpp teacher module available")
            
            # Check if llama-cpp-python is installed
            try:
                import llama_cpp
                print("✅ llama-cpp-python installed")
            except ImportError:
                print("⚠️  llama-cpp-python not installed")
                print("   Install with: pip install llama-cpp-python")
                return False
                
        except Exception as e:
            print(f"❌ Error loading GGUF teacher: {e}")
            return False
            
    elif backend == "hf":
        print("\n🤗 HuggingFace Configuration:")
        model_id = os.getenv("TEACHER_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        dtype = os.getenv("TEACHER_DTYPE", "bf16")
        
        print(f"✅ Model ID: {model_id}")
        print(f"✅ Data type: {dtype}")
        print("⚠️  This will download ~15GB on first run")
        
        try:
            from src.infra.teacher_local_hf import LocalHFTeacher
            print("✅ HuggingFace teacher module available")
        except Exception as e:
            print(f"❌ Error loading HF teacher: {e}")
            return False
    else:
        print(f"❌ Invalid TEACHER_BACKEND: {backend}")
        print("   Must be one of: openrouter, gguf, hf")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Teacher backend configuration looks good!")
    print("\nTo run training with this backend:")
    print("  python -m src.app.train_gold")
    print("\nOr use the full comparison script:")
    print("  bash scripts/run_full_comparison.sh")
    return True

if __name__ == "__main__":
    success = test_teacher_backend()
    sys.exit(0 if success else 1)
