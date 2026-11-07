"""Load the installed RIA modules."""
from __future__ import annotations
import os
from pathlib import Path


def load(name, namespace):
    root = Path(__file__).resolve().parents[4]
    base = Path(os.environ.get('SUBMISSION_RIA_SOURCE_DIR', root/'third_party/sources/ria'))
    path = base/'lib'/name
    if not path.is_file():
        raise FileNotFoundError('Install RIA first: python third_party/deit/setup_ria.py')
    content = path.read_bytes()
    exec(compile(content, str(path), 'exec'), namespace)
