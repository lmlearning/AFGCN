from pathlib import Path

import pytest

from Solver.af_input import read_af_input


def parse(tmp_path, text):
    path = tmp_path / "graph.af"
    path.write_text(text, encoding="utf-8")
    return read_af_input(path)


def test_blank_lines_comments_and_whitespace(tmp_path):
    assert parse(tmp_path, "\n  # comment\n p af 3\n\n1\t2\n 2  3\n") == (
        ["1", "2", "3"], [["1", "2"], ["2", "3"]]
    )


def test_isolated_arguments_and_self_attacks(tmp_path):
    assert parse(tmp_path, "p af 2\n1 1\n") == (["1", "2"], [["1", "1"]])


def test_empty_framework(tmp_path):
    assert parse(tmp_path, "p af 0\n") == ([], [])


@pytest.mark.parametrize("filename,count,extra_attacks", [
    ("testaf1.txt", 5, []),
    ("testaf2.txt", 8, []),
    ("testaf3.txt", 8, [["7", "8"], ["8", "7"]]),
])
def test_shipped_solver_examples(filename, count, extra_attacks):
    path = Path(__file__).resolve().parents[1] / "Solver" / filename
    arguments, attacks = read_af_input(path)
    assert arguments == [str(i) for i in range(1, count + 1)]
    assert attacks == [["1", "2"], ["2", "4"], ["4", "5"], ["5", "4"], ["5", "5"]] + extra_attacks


@pytest.mark.parametrize("text,message", [
    ("\n# comment\n", "missing"),
    ("1 2\n", "header"),
    ("p af 2\np af 2\n", "duplicate"),
    ("p\n", "expected"),
    ("p af -1\n", "nonnegative"),
    ("p af x\n", "integer"),
    ("p af 2\n1\n", "endpoints"),
    ("p af 2\n0 2\n", "outside"),
    ("p af 2\n1 3\n", "outside"),
])
def test_malformed_frameworks_report_context(tmp_path, text, message):
    with pytest.raises(ValueError, match="graph.af.*" + message):
        parse(tmp_path, text)
