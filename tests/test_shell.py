import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash adapter requires Bash")


def wrapper(*args):
    return subprocess.run([BASH, str(ROOT / "Solver/solver.sh"), *args],
                          env={**os.environ, "PYTHON": sys.executable}, capture_output=True, text=True)


def test_advertised_tasks_and_format_match_implementation():
    tasks = wrapper("--problems")
    assert tasks.returncode == 0
    assert set(tasks.stdout.strip().strip("[]").split(", ")) == set(json.loads((ROOT / "Solver/thresholds.json").read_text()))
    assert wrapper("--formats").stdout.strip() == "[i23]"


@pytest.mark.parametrize("args", [("-p",), ("--unknown",), ("-fo", "apx")])
def test_wrapper_rejects_invalid_invocations(args):
    assert wrapper(*args).returncode == 2


def test_wrapper_quotes_paths_and_propagates_solver_failure(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("dgl")
    path = tmp_path / "framework with spaces.af"
    path.write_text("p af 1\n")
    success = wrapper("-p", "DC-CO", "-f", str(path), "-a", "1")
    assert success.returncode == 0, success.stderr
    assert success.stdout.strip() == "YES"
    failure = wrapper("-p", "DC-CO", "-f", str(path), "-a", "2")
    assert failure.returncode == 2
