"""Execute every ```python block of the handbook chapters to prove the samples run.

Blocks are executed in order, one shared namespace per chapter (like a notebook), inside a
temporary working directory. Blocks fenced as ```python norun are skipped: they need external
services (databases, Kafka, HTTP APIs, servers) or are fragments shown for illustration.

Usage: python run_snippets.py [chapter.md ...]
"""

from __future__ import annotations

import os

# Containers often expose more vCPUs than their CPU quota. OpenMP/BLAS spin-waiting with too
# many threads then slows training by 100x (see handbook ch. 8.11). Pin native thread pools to 1
# and let joblib (n_jobs) provide the parallelism. Must happen before numpy/sklearn are imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
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


DEPRECATION = (FutureWarning, DeprecationWarning, PendingDeprecationWarning)


def run_chapter(path: Path) -> tuple[int, int, int, list[str], set[str]]:
    text = path.read_text(encoding="utf-8")
    ns: dict = {"__name__": "__handbook__"}
    ok = skipped = 0
    errors: list[str] = []
    deprecations: set[str] = set()
    for m in BLOCK.finditer(text):
        info, code = m.group(1).strip(), m.group(2)
        line = text[: m.start()].count("\n") + 1
        if "norun" in info:
            skipped += 1
            continue
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                exec(compile(code, f"{path.name}:{line}", "exec"), ns)
                ok += 1
            except Exception:
                errors.append(f"{path.name}:{line}\n{traceback.format_exc(limit=3)}")
            finally:
                plt.close("all")
        for w in caught:
            # chỉ báo cảnh báo trỏ vào code của handbook (thư viện tự cảnh báo nội bộ thì bỏ qua)
            if issubclass(w.category, DEPRECATION) and "site-packages" not in str(w.filename):
                deprecations.add(f"{path.name}:{line} {w.category.__name__}: {str(w.message)[:160]}")
        dt = time.perf_counter() - t0
        if dt > 30 or os.environ.get("HANDBOOK_PROFILE"):
            print(f"  block {path.name}:{line} {dt:.0f}s", flush=True)
    return ok, skipped, len(errors), errors, deprecations


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] or sorted((ROOT / "chapters").glob("[0-9][0-9]_*.md"))
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    total_err = 0
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        for f in files:
            t0 = time.perf_counter()
            ok, skipped, n_err, errors, deps = run_chapter(f)
            total_err += n_err
            print(f"{f.name:40s} ok={ok:3d} skipped={skipped:2d} errors={n_err} "
                  f"deprecations={len(deps)} ({time.perf_counter() - t0:.0f}s)")
            for e in errors:
                print("   ", e.replace("\n", "\n    "))
            for d in sorted(deps):
                print("    [deprecation]", d)
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
