"""Every pipeline step runs from the repo's work/ folder; raw downloads live in data/ and the finished pages in docs/."""
import os
from pathlib import Path

# The rating fits solve small dense systems; multithreaded BLAS makes them ~40x slower on many-core machines.
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
ROOT = Path(__file__).resolve().parent.parent
for d in ('data', 'work', 'docs'):
    (ROOT / d).mkdir(exist_ok=True)
os.chdir(ROOT / 'work')
