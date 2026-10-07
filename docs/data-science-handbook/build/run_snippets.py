"""Execute every ```python block of the handbook chapters to prove the samples run.

Blocks are executed in order, one shared namespace per chapter (like a notebook), inside a
temporary working directory. Blocks fenced as ```python norun are skipped: they need external
services (databases, Kafka, HTTP APIs, servers) or are fragments shown for illustration.

Usage: python run_snippets.py [chapter.md ...]
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code" / "src"))
BLOCK = re.compile(r"^```python([^\n]*)\n(.*?)^```", re.S | re.M)


def run_chapter(path: Path) -> tuple[int, int, int, list[str]]:
    text = path.read_text(encoding="utf-8")
    ns: dict = {"__name__": "__handbook__"}
    ok = skipped = 0
    errors: list[str] = []
    for m in BLOCK.finditer(text):
        info, code = m.group(1).strip(), m.group(2)
        line = text[: m.start()].count("\n") + 1
        if "norun" in info:
            skipped += 1
            continue
        t0 = time.perf_counter()
        try:
            exec(compile(code, f"{path.name}:{line}", "exec"), ns)
            ok += 1
        except Exception:
            errors.append(f"{path.name}:{line}\n{traceback.format_exc(limit=3)}")
        finally:
            plt.close("all")
        dt = time.perf_counter() - t0
        if dt > 30:
            print(f"  slow block {path.name}:{line} {dt:.0f}s")
    return ok, skipped, len(errors), errors


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] or sorted((ROOT / "chapters").glob("[0-9][0-9]_*.md"))
    warnings.filterwarnings("ignore")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    total_err = 0
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        for f in files:
            t0 = time.perf_counter()
            ok, skipped, n_err, errors = run_chapter(f)
            total_err += n_err
            print(f"{f.name:40s} ok={ok:3d} skipped={skipped:2d} errors={n_err} ({time.perf_counter() - t0:.0f}s)")
            for e in errors:
                print("   ", e.replace("\n", "\n    "))
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
