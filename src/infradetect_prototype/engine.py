"""Subprocess adapter to the existing nine-stage integrated scene pipeline.

Running from the delivery root selects its self-contained ``infradetect`` package.
The active Python interpreter is reused, so the caller controls the environment
without modifying it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from .contract import SceneRequest


def run_scene(request: SceneRequest, *, check_only: bool = False,
              quiet: bool = False) -> dict:
    """Validate or execute a scene; propagate pipeline errors unchanged.

    The pipeline itself writes a manifest snapshot, preflight, stage status,
    evidence maps, and report. No output directory is silently replaced.
    """

    request = request.checked(will_write=not check_only)
    command = [sys.executable, "-m", "infradetect", "--manifest",
               str(request.manifest), "--output", str(request.output),
               "--project-root", str(request.project_root)]
    if check_only:
        command.append("--check")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(request.project_root), environment.get("PYTHONPATH", "")) if part
    )
    # Normal runs stream progress. The synthetic smoke check suppresses its
    # verbose report; errors still reach stderr and all artifacts stay on disk.
    if not check_only:
        subprocess.run(command, cwd=request.project_root, env=environment, check=True,
                       stdout=subprocess.DEVNULL if quiet else None)
        from .report import summarize_run

        return summarize_run(request.output)
    completed = subprocess.run(command, cwd=request.project_root, env=environment,
                               text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)
