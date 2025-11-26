"""Utility helpers for managing uploaded files and GA outputs."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

from .constants import ALLOWED_EXTENSIONS

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE_ROOT / "data"
GA_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "ga_runs"
PLOT_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "plots"
UPLOAD_CACHE_ROOT = WORKSPACE_ROOT / "uploads"


def ensure_directories() -> None:
    """Ensure core directory structure exists."""
    for path in (DATA_ROOT, GA_OUTPUT_ROOT, PLOT_OUTPUT_ROOT, UPLOAD_CACHE_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def write_upload(content: bytes, original_name: str, model_value: str) -> Tuple[Path, str]:
    """Persist an uploaded file to the correct data sub-directory.

    Returns a tuple of the saved path and the sanitized filename used for GA runs.
    """
    ensure_directories()
    extension = original_name.rsplit(".", 1)[-1].lower()
    safe_name = original_name.replace(" ", "_")
    if not allowed_file(original_name):
        safe_name = f"upload_{uuid.uuid4().hex}.{extension}"

    subdir = "Ellipsoids" if model_value == "ellipsoids" else "hollowTubes"
    target_dir = DATA_ROOT / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / safe_name
    target_path.write_bytes(content)
    return target_path, safe_name


def build_ga_output_paths(model_value: str, file_stem: str) -> Tuple[Path, Path]:
    """Return the GA output directory and violin plot path for a given run."""
    ensure_directories()
    model_prefix = "Ellipsoids" if model_value == "ellipsoids" else "hollowTubes"
    run_root = GA_OUTPUT_ROOT / f"{model_prefix}_{file_stem}"
    plot_path = run_root / "GArun_0" / "final_generation_violin_plots.png"
    return run_root, plot_path


def resolve_existing_model_path(model_filename: str) -> Optional[Path]:
    """Locate an existing model file from various fallback directories."""
    candidates = [
        WORKSPACE_ROOT / "models" / model_filename,
        WORKSPACE_ROOT / ".." / "01" / "models" / model_filename,
        WORKSPACE_ROOT / ".." / "models" / model_filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None
