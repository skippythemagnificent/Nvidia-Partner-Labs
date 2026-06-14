"""Repo-root conftest.

Guarantees the project root is on `sys.path` for every pytest run, so
`import shared` works regardless of pytest's import mode or the test's location.
Plain `python` and `task` commands import `shared` via the repo-root cwd, but
pytest's prepend import mode puts the test directory on the path instead of the
root — without this shim, `import shared` fails under pytest.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
