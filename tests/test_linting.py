import subprocess


def test_make_lint_runs_without_errors():
    result = subprocess.run(["make", "lint"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
