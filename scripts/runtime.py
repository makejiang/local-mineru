"""Single-process inference runtime for local-mineru."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


# --- Make the package dir importable, then resolve the mineru_ov package ----
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _resolve_mineru_ov_path() -> str:
    """Resolve the ``mineru_ov`` package directory.

    Prefer an explicit ``LOCAL_MINERU_OV_PATH`` and otherwise use the package
    bundled next to this module.
    """
    env_path = os.environ.get("LOCAL_MINERU_OV_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parent.parent / "mineru_ov")

    for p in candidates:
        if p.is_dir() and (p / "__init__.py").exists():
            return str(p)

    raise FileNotFoundError(
        "mineru_ov package not found. Tried: "
        + ", ".join(str(p) for p in candidates)
        + ". Set LOCAL_MINERU_OV_PATH or ensure mineru_ov/ is bundled with the skill."
    )


try:
    MINERU_OV_PATH = _resolve_mineru_ov_path()
    sys.path.insert(0, str(Path(MINERU_OV_PATH).parent))
except FileNotFoundError as e:
    print(f"Fatal: {e}", file=sys.stderr)
    sys.exit(1)

from model_download import (
    ensure_models,
    load_model_infos,
    validate_model_dir,
)

STATE_STARTING = "starting"
STATE_DOWNLOADING = "downloading"
STATE_LOADING = "loading"
STATE_RUNNING = "running"
STATE_ERROR = "error"

OPENVINO_ROOT = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".openvino"
MODELS_ROOT = OPENVINO_ROOT / "models"

_HERE = Path(__file__).resolve().parent
INFO_JSON = _HERE / "info.json" if (_HERE / "info.json").exists() else _HERE.parent / "info.json"
MODEL_INFOS = load_model_infos(INFO_JSON)
MODEL_INFO = MODEL_INFOS[0]
MODEL_ID = MODEL_INFO.model_id
# info.json's required_files are nested under the top-level model dir; the
# pipeline's models_dir is that top-level dir (e.g. .../mineru-ov-models).
MODEL_BASE_DIR = MODELS_ROOT / MODEL_INFO.dir_name
REQUIRED_MODEL_FILES = MODEL_INFO.required_files

DEFAULT_DEVICE = "GPU"
DEFAULT_PRECISION = "int4"

os.environ.setdefault("OPENVINO_TELEMETRY_OPT_OUT", "1")

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream is not None and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

LOG_LOCK = threading.Lock()


def _normalize_log_path(log_path: str | None) -> Path | None:
    if not log_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = f"~/.openvino/log/mineru-runtime-py-{timestamp}.log"

    path = Path(log_path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _log_message(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with LOG_LOCK:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"[{timestamp}] [runtime pid={os.getpid()}] {message}\n")
    except OSError:
        pass


def _report_download_message(log_path: Path | None, message: str) -> None:
    print(f"[runtime] {message}", flush=True)
    _log_message(log_path, message)


def _ensure_required_model(log_path: Path | None = None) -> None:
    """Download the model to MODEL_BASE_DIR if not already valid."""
    validation = validate_model_dir(MODEL_BASE_DIR, REQUIRED_MODEL_FILES)
    if validation.ok:
        return

    def _logger(message: str):
        _report_download_message(log_path, message)

    ensure_models(MODEL_INFOS, MODELS_ROOT, logger=_logger)

    validation = validate_model_dir(MODEL_BASE_DIR, REQUIRED_MODEL_FILES)
    if not validation.ok:
        raise RuntimeError(
            f"required model download did not complete: {MODEL_BASE_DIR} ({validation.reason})"
        )


# ---------------------------------------------------------------------------
# MinerU OpenVINO pipeline helpers
# ---------------------------------------------------------------------------
def _load_pipeline(device: str, precision: str) -> Any:
    """Build a ``MinerUPipeline`` from the mineru_ov package on the given device.

    The ContinuousBatching backend is on by default; set
    ``LOCAL_MINERU_DISABLE_BATCH=1`` to fall back to the single-page backend.
    """
    from mineru_ov.config import OVConfig
    from mineru_ov.pipeline import MinerUPipeline

    use_batch = os.environ.get("LOCAL_MINERU_DISABLE_BATCH", "").strip() not in ("1", "true", "True")
    max_concurrent = int(os.environ.get("LOCAL_MINERU_MAX_CONCURRENT", "8"))

    cfg = OVConfig(
        device=device,
        precision=precision,
        models_dir=str(MODEL_BASE_DIR),
        backend_type="vlm",
        use_batch_backend=use_batch,
        max_concurrent=max_concurrent,
        performance_hint="LATENCY",
    )
    cfg.validate()
    return MinerUPipeline(cfg)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class Runtime:
    def __init__(self, log_path: Path | None = None) -> None:
        self.runtime_lock = threading.Lock()
        self.pipeline: Any = None
        self.loaded_model_id: str | None = None
        self.loaded_device: str | None = None
        self.log_path = log_path
        self.state: str = STATE_STARTING
        self.init_error: str = ""

    def log(self, message: str) -> None:
        _log_message(self.log_path, message)

    def initialize(self) -> None:
        try:
            with self.runtime_lock:
                self.state = STATE_DOWNLOADING
            self.log("initialization: downloading required model")
            _ensure_required_model(self.log_path)

            with self.runtime_lock:
                self.state = STATE_LOADING
            self.log("initialization: loading OpenVINO MinerU pipeline")

            pipeline = _load_pipeline(DEFAULT_DEVICE, DEFAULT_PRECISION)

            with self.runtime_lock:
                self.pipeline = pipeline
                self.loaded_model_id = MODEL_ID
                self.loaded_device = DEFAULT_DEVICE
                self.state = STATE_RUNNING
            self.log(f"initialization: pipeline ready on device={DEFAULT_DEVICE}")
        except Exception:
            error_text = traceback.format_exc()
            with self.runtime_lock:
                self.init_error = error_text
                self.state = STATE_ERROR
            self.log(f"initialization failed:\n{error_text}")

    def _do_parse(self, input_path: str, output_dir: str) -> dict:
        t0 = time.time()
        in_path = Path(input_path).resolve()
        if not in_path.exists():
            return {
                "ok": True,
                "success": False,
                "error_code": "BAD_INPUT",
                "error": f"input path does not exist: {in_path}",
            }
        out_dir = Path(output_dir).resolve() if output_dir else (in_path.parent / "output")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.log(f"parsing: {in_path} -> {out_dir}")
        markdown, page_count = self.pipeline.run(
            input_path=str(in_path),
            output_dir=str(out_dir),
        )
        parse_time_s = time.time() - t0
        self.log(f"parse done: {page_count} pages in {parse_time_s:.1f}s")

        return {
            "ok": True,
            "success": True,
            "markdown": markdown,
            "pages": page_count,
            "md_file": str(out_dir / f"{in_path.stem}.md"),
            "output_dir": str(out_dir),
            "parse_time_s": parse_time_s,
            "device": self.loaded_device,
            "model_id": self.loaded_model_id,
        }

    def dispatch(self, msg: dict) -> dict:
        op = msg.get("op")
        self.log(f"dispatching op={op!r}")
        if op == "parse":
            input_path = msg.get("input_path")
            if not isinstance(input_path, str) or not input_path.strip():
                return {
                    "ok": False,
                    "success": False,
                    "error_code": "BAD_INPUT",
                    "error": "input_path must be a non-empty string",
                }
            with self.runtime_lock:
                state = self.state
                init_error = self.init_error
                pipeline = self.pipeline
            if state != STATE_RUNNING or pipeline is None:
                self.log(f"parse rejected: state={state}")
                reply = {
                    "ok": False,
                    "success": False,
                    "state": state,
                    "error": f"runtime not ready: {state}",
                }
                if state == STATE_ERROR and init_error:
                    reply["error"] = init_error
                return reply
            try:
                reply = self._do_parse(input_path, msg.get("output_dir", ""))
                reply["state"] = STATE_RUNNING
                self.log(
                    f"parse completed: success={reply.get('success')} "
                    f"md_file={reply.get('md_file')}"
                )
                return reply
            except Exception:
                error_text = traceback.format_exc()
                self.log(f"parse failed:\n{error_text}")
                return {
                    "ok": False,
                    "success": False,
                    "state": STATE_RUNNING,
                    "error": error_text,
                }
        self.log(f"unknown operation received: {op!r}")
        return {"ok": False, "error": f"unknown op: {op!r}"}
