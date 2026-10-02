"""Every pipeline step runs from the repo's work/ folder; raw downloads live in data/ and the finished page in docs/."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for d in ('data', 'work', 'docs'):
    (ROOT / d).mkdir(exist_ok=True)
os.chdir(ROOT / 'work')
