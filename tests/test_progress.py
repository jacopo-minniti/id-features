import ast
from pathlib import Path


def test_batch_progress_uses_explicit_completed_counter() -> None:
    """A disabled tqdm counter does not advance, so batch logs need their own count."""
    module = ast.parse(Path("src/id_features/experiments.py").read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
    assert "completed_measurements" in names
