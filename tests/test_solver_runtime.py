from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")

from Solver.solver import AFGCNModel, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_cpu_thread():
    torch.set_num_threads(1)


def test_checkpoint_inference_disables_dropout_and_preserves_node_axis():
    model = AFGCNModel(128, 128, 128, 1)
    load_checkpoint(model, ROOT / "Solver/DC-CO.pth")
    assert not model.training
    graph = dgl.graph(([0], [0]), num_nodes=1)
    inputs = torch.ones(1, 128)
    with torch.no_grad():
        first, second = model(graph, inputs), model(graph, inputs)
    assert first.shape == (1,)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def run_solver(path, *extra):
    return subprocess.run([
        sys.executable, str(ROOT / "Solver/solver.py"), "--filepath", str(path),
        "--task", "DC-CO", "--argument", "1", *extra,
    ], capture_output=True, text=True, timeout=60)


def test_single_argument_neural_path_and_repeatability(tmp_path):
    path = tmp_path / "self attack.af"
    path.write_text("p af 1\n1 1\n")
    first, second = run_solver(path), run_solver(path)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert first.stdout == second.stdout
    assert first.stdout.strip() in {"YES", "NO"}


def test_grounded_example_from_another_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_solver(ROOT / "Solver/testaf1.txt")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "YES"


def test_invalid_query_reports_actionable_error(tmp_path):
    path = tmp_path / "empty.af"
    path.write_text("p af 0\n")
    result = run_solver(path)
    assert result.returncode == 2
    assert "not declared" in result.stderr
    assert "Traceback" not in result.stderr
