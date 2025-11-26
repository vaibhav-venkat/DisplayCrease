"""Background GA runner that exposes thread-safe state for Dash callbacks."""
from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

CSV_START_TOKEN = "__CREASE_CSV_BEGIN__"
CSV_END_TOKEN = "__CREASE_CSV_END__"


@dataclass
class GAState:
    status: str = "idle"  # idle | running | completed | failed
    logs: str = ""
    last_error: Optional[str] = None
    csv_results: Optional[str] = None


class GeneticAlgorithmManager:
    def __init__(self) -> None:
        self._state = GAState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None

    def get_state(self) -> GAState:
        with self._lock:
            return GAState(
                status=self._state.status,
                logs=self._state.logs,
                last_error=self._state.last_error,
                csv_results=self._state.csv_results,
            )

    def reset(self) -> None:
        """Reset state and kill any running processes."""
        with self._lock:
            # Kill existing process if running
            if self._process and self._process.poll() is None:
                try:
                    self._process.terminate()
                    # Give it a moment to terminate gracefully
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate
                        self._process.kill()
                        self._process.wait()
                except Exception:
                    # Process might already be dead, that's fine
                    pass
                finally:
                    self._process = None
            
            # Reset state
            self._state = GAState()

    def start(
        self,
        filename: str,
        model_value: str,
        working_dir: Path,
    ) -> None:
        # Always reset/kill existing processes before starting
        self.reset()
        
        with self._lock:
            self._state = GAState(status="running", logs="Starting Genetic Algorithm...\n")

        script = working_dir / "GA_py_script_modular.py"
        model_arg = "Ellipsoids" if model_value == "ellipsoids" else "hollowTubes"
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        sample_name = Path(filename).stem
        sample_name = sample_name.replace("_orig_datayz", "").replace("_datayz", "").replace("Ellipsoids_test_", "")

        def _run() -> None:
            process = subprocess.Popen(
                [
                    "python",
                    str(script),
                    filename,
                    model_arg,
                ],
                cwd=str(working_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            
            # Store process reference for potential termination
            with self._lock:
                self._process = process

            collected_logs = []
            collecting_csv = False
            csv_lines: list[str] = []
            try:
                for line in process.stdout:  # type: ignore[attr-defined]
                    stripped = line.rstrip("\n")
                    if stripped == CSV_START_TOKEN:
                        collecting_csv = True
                        csv_lines = []
                        continue
                    if stripped == CSV_END_TOKEN:
                        collecting_csv = False
                        csv_content = "\n".join(csv_lines)
                        with self._lock:
                            self._state.csv_results = csv_content
                        continue
                    if collecting_csv:
                        csv_lines.append(stripped)
                        continue

                    collected_logs.append(stripped)
                    with self._lock:
                        self._state.logs = "\n".join(collected_logs)
                return_code = process.wait()
                with self._lock:
                    if return_code == 0:
                        self._state.status = "completed"
                    else:
                        self._state.status = "failed"
                        self._state.last_error = f"GA script exited with code {return_code}"
            except Exception as exc:  # pylint: disable=broad-except
                with self._lock:
                    self._state.status = "failed"
                    self._state.last_error = str(exc)
            finally:
                if process.stdout:
                    process.stdout.close()
                # Clear process reference when done
                with self._lock:
                    if self._process == process:
                        self._process = None

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()


GA_MANAGER = GeneticAlgorithmManager()
