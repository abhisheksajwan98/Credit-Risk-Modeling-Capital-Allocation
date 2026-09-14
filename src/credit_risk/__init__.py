"""Graph-based credit risk and reinforcement-learning lending decision system.

See ``docs/PROJECT_SPEC.md`` for the specification this package implements.
"""

from __future__ import annotations

import os

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# OpenMP runtime coexistence (Windows, conda + pip)
# ---------------------------------------------------------------------------
# On the reference environment numpy/scipy/LightGBM come from conda and are linked against
# Intel MKL's OpenMP (`libiomp5md.dll`), while PyTorch is a pip wheel that bundles its own copy.
# When the second one initialises, the Intel runtime aborts the process with "OMP: Error #15".
#
# This flag is Intel's documented (if grudging) escape hatch and is what essentially every
# conda + pip-torch installation on Windows uses. The stated risk is that two OpenMP runtimes
# competing for threads can degrade performance; in this project the two never run nested
# parallel regions -- LightGBM training and torch training happen in separate phases -- so the
# practical exposure is low.
#
# The clean alternatives, for the record: build the whole environment from one channel, or use
# openblas-linked numpy/scipy instead of MKL. Both were judged not worth the disruption here.
# Set the variable yourself before importing to override this default.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
