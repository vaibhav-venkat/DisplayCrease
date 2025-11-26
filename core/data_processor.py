"""Data ingestion and transformation utilities for scattering matrices."""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import DEFAULT_DATA_CONFIG


META_PATTERNS = {
    "q_min_exp": re.compile(r"q[_\s-]*min(?:_exp)?[:=\s]+(-?\d+\.?\d*)", re.IGNORECASE),
    "q_max_exp": re.compile(r"q[_\s-]*max(?:_exp)?[:=\s]+(-?\d+\.?\d*)", re.IGNORECASE),
    "phi_min": re.compile(r"phi[_\s-]*min[:=\s]+(-?\d+\.?\d*)", re.IGNORECASE),
    "phi_max": re.compile(r"phi[_\s-]*max[:=\s]+(-?\d+\.?\d*)", re.IGNORECASE),
    "is_log_transformed": re.compile(r"log[_\s-]*transformed[:=\s]+(true|false|yes|no|1|0)", re.IGNORECASE),
}


@dataclass
class MatrixStatistics:
    rows: int
    columns: int
    min_value: float
    max_value: float
    mean: float
    std: float
    phi_min: float
    phi_max: float
    q_min_exp: float
    q_max_exp: float
    is_log_transformed: bool

    def to_dict(self) -> Dict[str, float]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean": self.mean,
            "std": self.std,
            "phi_min": self.phi_min,
            "phi_max": self.phi_max,
            "q_min_exp": self.q_min_exp,
            "q_max_exp": self.q_max_exp,
            "is_log_transformed": self.is_log_transformed,
        }


@dataclass
class ProcessedMatrix:
    matrix: np.ndarray
    statistics: MatrixStatistics
    metadata: Dict[str, float]

    def to_dataframe(self) -> pd.DataFrame:
        phi_values = np.linspace(
            self.statistics.phi_min,
            self.statistics.phi_max,
            self.matrix.shape[1],
        )
        q_exp_values = np.linspace(
            self.statistics.q_min_exp,
            self.statistics.q_max_exp,
            self.matrix.shape[0],
        )

        records: List[Dict[str, float]] = []
        for i, q_exp in enumerate(q_exp_values):
            for j, phi in enumerate(phi_values):
                records.append(
                    {
                        "q_index": i,
                        "phi_index": j,
                        "phi_deg": float(phi),
                        "q_exp": float(q_exp),
                        "q_real": float(math.pow(10.0, q_exp)),
                        "intensity": float(self.matrix[i, j]),
                    }
                )
        return pd.DataFrame.from_records(records)


class MatrixProcessingError(Exception):
    """Raised when uploaded content cannot be parsed."""


def _parse_metadata(line: str) -> Dict[str, float]:
    line = line.strip("#/ ")
    metadata: Dict[str, float] = {}
    for key, pattern in META_PATTERNS.items():
        match = pattern.search(line)
        if match:
            value = match.group(1)
            if key == "is_log_transformed":
                metadata[key] = value.lower() in {"true", "yes", "1"}
            else:
                metadata[key] = float(value)
    return metadata


def _normalize_delimiters(line: str) -> List[str]:
    if "," in line:
        return [token.strip() for token in line.split(",") if token.strip()]
    # Fallback to whitespace splitting
    return [token for token in re.split(r"\s+", line) if token]


def parse_matrix_text(content: str, user_config: Optional[Dict[str, float]] = None) -> ProcessedMatrix:
    """Parse raw text containing a scattering intensity matrix."""
    if not content.strip():
        raise MatrixProcessingError("Uploaded file is empty")

    metadata: Dict[str, float] = {}
    data_rows: List[List[float]] = []

    for raw_line in io.StringIO(content):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("//"):
            metadata.update(_parse_metadata(line))
            continue
        try:
            tokens = _normalize_delimiters(line)
            if not tokens:
                continue
            data_rows.append([float(token) for token in tokens])
        except ValueError as exc:
            raise MatrixProcessingError(f"Invalid numeric value in line: '{line}'") from exc

    if not data_rows:
        raise MatrixProcessingError("No numeric data found in file")

    num_cols = len(data_rows[0])
    if any(len(row) != num_cols for row in data_rows):
        raise MatrixProcessingError("Inconsistent column counts detected")

    matrix = np.array(data_rows, dtype=float)
    stats = _compute_statistics(matrix, metadata, user_config)
    return ProcessedMatrix(matrix=matrix, statistics=stats, metadata=metadata)


def _compute_statistics(
    matrix: np.ndarray,
    metadata: Dict[str, float],
    user_config: Optional[Dict[str, float]] = None,
) -> MatrixStatistics:
    config = {**DEFAULT_DATA_CONFIG}
    if metadata:
        config.update({k: v for k, v in metadata.items() if k in config})
    if user_config:
        config.update({k: v for k, v in user_config.items() if v is not None})

    flat = matrix.flatten()
    return MatrixStatistics(
        rows=matrix.shape[0],
        columns=matrix.shape[1],
        min_value=float(np.min(flat)),
        max_value=float(np.max(flat)),
        mean=float(np.mean(flat)),
        std=float(np.std(flat)),
        phi_min=float(config["phi_min"]),
        phi_max=float(config["phi_max"]),
        q_min_exp=float(config["q_min_exp"]),
        q_max_exp=float(config["q_max_exp"]),
        is_log_transformed=bool(config["is_log_transformed"]),
    )
