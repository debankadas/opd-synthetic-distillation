import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_LOG_FH = None  # type: ignore[var-annotated]
_LOG_PATH: Optional[Path] = None


def init_event_log(output_dir: str = "runs", filename: Optional[str] = None) -> str:
    """
    Initialize a JSONL log file under output_dir. If filename is None, create
    gold_train_YYYYmmdd_HHMMSS.jsonl. Returns absolute path to the log file.
    """
    global _LOG_FH, _LOG_PATH
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gold_train_{ts}.jsonl"
    _LOG_PATH = outdir / filename
    _LOG_FH = _LOG_PATH.open("a", encoding="utf-8")
    return str(_LOG_PATH.resolve())


def log_event(event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"event": event}
    for k, v in fields.items():
        if is_dataclass(v):
            payload[k] = asdict(v)
        else:
            payload[k] = v
    line = json.dumps(payload, ensure_ascii=False)
    # stdout
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    # file
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()

