"""Pytest config — fix FAISS + PyTorch thread conflicts on macOS."""

import os

# without these, FAISS and PyTorch fight over OpenMP on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
