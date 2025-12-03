from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Optional, List

try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
except Exception:  # pragma: no cover - optional imports
    snapshot_download = None
    hf_hub_download = None
    list_repo_files = None


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "_")


def ensure_student_model_available(local_or_id: str, cache_root: str = "models/hf") -> str:
    """
    Ensure a Hugging Face model is available locally.
    - If local_or_id is a directory path and exists, return it.
    - Otherwise, treat it as a repo id and snapshot_download into cache_root/<sanitized_id>.
    Returns a local directory path.
    """
    p = Path(local_or_id)
    if p.exists() and p.is_dir():
        return str(p)
    # Optional: if an explicit Xet command template is provided, use it; otherwise rely on huggingface_hub (hf_xet accelerates under the hood)
    if os.getenv("XET_HF_CMD"):
        return _download_student_with_xet(local_or_id, cache_root)
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub not available to download student model")
    target_dir = Path(cache_root) / _sanitize_repo_id(local_or_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    # Download (or re-use cache) into target_dir. hf_xet (>=0.32.0) speeds this automatically when available.
    snapshot_download(repo_id=local_or_id, local_dir=str(target_dir))
    return str(target_dir)


def ensure_teacher_gguf_available(
    gguf_path: str,
    repo_id_env: str = "TEACHER_GGUF_HF_REPO_ID",
    filename_env: str = "TEACHER_GGUF_FILENAME",
    cache_root: str = "models",
) -> str:
    """
    Ensure a GGUF model file exists locally at gguf_path.
    If not found, try downloading from HF using env vars:
      - TEACHER_GGUF_HF_REPO_ID (e.g., TheBloke/Llama-2-7B-GGUF)
      - TEACHER_GGUF_FILENAME (e.g., llama-2-7b.Q4_K_M.gguf)
    Saves into cache_root and returns full path.
    """
    dest = Path(gguf_path)
    if dest.exists() and dest.is_file():
        return str(dest)
    # Optional: support Xet CLI if variables provided, else use huggingface_hub
    if os.getenv("XET_GGUF_CMD") or os.getenv("XET_GGUF_SOURCE"):
        return _download_gguf_with_xet(gguf_path)
    repo_id = os.getenv(repo_id_env)
    filename = os.getenv(filename_env)
    if not repo_id or not filename:
        raise FileNotFoundError(
            f"GGUF not found at {gguf_path} and env {repo_id_env}/{filename_env} not set to download."
        )
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not available to download GGUF teacher model")
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    local_file = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=cache_root)
    # Move/rename to desired path if different
    if Path(local_file) != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        Path(local_file).replace(dest)
    return str(dest)


def _score_gguf_filename(name: str) -> int:
    # Higher is better; prefer Q4_K_M variants, then other Q4_K, then other .gguf
    n = name.lower()
    if not n.endswith('.gguf'):
        return -1
    score = 1
    if 'q4_k' in n:
        score += 5
    if 'q4_k_m' in n or 'q4_k_m' in n.replace('-', '_'):
        score += 10
    if '7b' in n:
        score += 2
    return score


def ensure_teacher_gguf_from_repo(
    repo_id: str,
    filename: Optional[str] = None,
    cache_root: str = "models/gguf",
) -> str:
    """
    Ensure a GGUF file from a HF model repo is available locally.
    - If filename is provided, download that file.
    - Else, list repo files and pick the best .gguf by a simple heuristic.
    Returns local filepath.
    """
    # Proceed via huggingface_hub; hf_xet (>=0.32.0) accelerates under the hood when available
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not available to download GGUF teacher model")
    dest_dir = Path(cache_root) / _sanitize_repo_id(repo_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_file: Optional[str] = filename if filename else None
    if target_file is None:
        if list_repo_files is None:
            raise RuntimeError("Cannot list repo files; upgrade huggingface_hub or specify TEACHER_GGUF_FILENAME")
        files: List[str] = list_repo_files(repo_id)
        # pick best .gguf
        candidates = sorted(files, key=_score_gguf_filename, reverse=True)
        candidates = [f for f in candidates if f.lower().endswith('.gguf')]
        if not candidates:
            raise FileNotFoundError(f"No .gguf files found in repo {repo_id}")
        target_file = candidates[0]
    local_file = hf_hub_download(repo_id=repo_id, filename=target_file, local_dir=str(dest_dir))
    return str(local_file)


def ensure_teacher_gguf_simple(
    teacher_spec: str,
    preferred_quant: str = "Q4_K_M",
    cache_root: str = "models/gguf",
) -> str:
    """
    Single-arg convenience:
      - If teacher_spec is a local .gguf path and exists, return it.
      - Else treat it as a repo id or HF URL; list and choose best .gguf (preferring preferred_quant), download, return local path.
    """
    p = Path(teacher_spec)
    if p.suffix.lower() == ".gguf" and p.exists():
        return str(p)
    repo_id = _parse_hf_repo_id(teacher_spec)
    # If filename inferred, prefer preferred_quant
    if list_repo_files is None or hf_hub_download is None:
        raise RuntimeError("huggingface_hub not available to download GGUF teacher model")
    files: List[str] = list_repo_files(repo_id)
    candidates = [f for f in files if f.lower().endswith('.gguf')]
    if not candidates:
        raise FileNotFoundError(f"No .gguf files found in repo {repo_id}")
    # Rank: contains preferred_quant highest, then _score_gguf_filename
    pref = preferred_quant.lower().replace('-', '_') if preferred_quant else ""
    def rank(name: str) -> tuple[int, int]:
        name_l = name.lower().replace('-', '_')
        has_pref = 1 if (pref and pref in name_l) else 0
        return (has_pref, _score_gguf_filename(name))
    best = sorted(candidates, key=rank, reverse=True)[0]
    dest_dir = Path(cache_root) / _sanitize_repo_id(repo_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_file = hf_hub_download(repo_id=repo_id, filename=best, local_dir=str(dest_dir))
    return str(local_file)


def _download_student_with_xet(repo_or_path: str, cache_root: str) -> str:
    # If a local dir is passed, just use it
    p = Path(repo_or_path)
    if p.exists() and p.is_dir():
        return str(p)
    # Require a command template to be provided via env for safety
    # Example: XET_HF_CMD='xet mount hf://{repo_id} {dest_dir}' OR 'xet cp -r hf://{repo_id} {dest_dir}'
    cmd_tpl = os.getenv("XET_HF_CMD")
    if not cmd_tpl:
        raise RuntimeError("HF_BACKEND=xet but XET_HF_CMD not set. Provide a command template with {repo_id} and {dest_dir}.")
    dest_dir = Path(cache_root) / _sanitize_repo_id(repo_or_path)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = cmd_tpl.format(repo_id=repo_or_path, dest_dir=str(dest_dir))
    _run_shell(cmd)
    if not dest_dir.exists() or not any(dest_dir.iterdir()):
        raise FileNotFoundError(f"Xet did not populate destination: {dest_dir}")
    return str(dest_dir)


def _download_gguf_with_xet(gguf_path: str) -> str:
    dest = Path(gguf_path)
    if dest.exists():
        return str(dest)
    # Either provide a full cp command template or a source and we will build a cp command
    cmd_tpl = os.getenv("XET_GGUF_CMD")
    src = os.getenv("XET_GGUF_SOURCE")
    if not cmd_tpl and not src:
        raise RuntimeError("HF_BACKEND=xet but neither XET_GGUF_CMD nor XET_GGUF_SOURCE is set.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cmd_tpl:
        # Default cp template
        cmd_tpl = "xet cp {src} {dest}"
    cmd = cmd_tpl.format(src=src, dest=str(dest)) if "{src}" in cmd_tpl else cmd_tpl.format(dest=str(dest))
    _run_shell(cmd)
    if not dest.exists():
        raise FileNotFoundError(f"Xet did not create GGUF file at: {dest}")
    return str(dest)


def _run_shell(cmd: str) -> None:
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nstdout: {proc.stdout.decode()}\nstderr: {proc.stderr.decode()}")
def _parse_hf_repo_id(spec: str) -> str:
    """Accept a HF repo id like 'org/name' or a full URL 'https://huggingface.co/org/name'."""
    if spec.startswith("http://") or spec.startswith("https://"):
        # Strip scheme and host
        try:
            parts = spec.split("huggingface.co/")[1]
        except IndexError:
            raise ValueError(f"Unrecognized HF URL: {spec}")
        repo_id = parts.strip("/")
        # Remove 'tree/main' or similar suffix if present
        repo_id = repo_id.split("/resolve/")[0].split("/blob/")[0].split("/tree/")[0]
        return repo_id
    return spec
