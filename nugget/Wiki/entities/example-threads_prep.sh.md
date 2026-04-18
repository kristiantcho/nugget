---
type: entity
status: draft
sources:
  - ../../nugget/examples/threads_prep.sh
updated: 2026-04-18
---

# threads_prep.sh

## Purpose
Configure OpenMP and MKL threading environment variables to ensure single-threaded execution of linear algebra libraries for reproducible and memory-efficient parallel computations across multiple GPU devices.

## What it does
Sets three critical threading environment variables to 1:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

This configuration:
- Disables multi-threaded execution within numpy/scipy/PyTorch linear algebra operations
- Prevents thread pool oversubscription when running multiple parallel processes (e.g., DataLoader workers, distributed training)
- Ensures memory usage stays within bounds when launching many workers
- Makes random seed behavior reproducible across runs

## Usage
Source this script before running example scripts that use parallel workers:

```bash
source ./threads_prep.sh
python train_signal_only_llr.py
```

Or inline at runtime:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python train_chargenet.py
```

## When to use
Required for:
- **DataLoader training**: Scripts using PyTorch DataLoader with num_workers > 1
  - train_chargenet.py (4 workers)
  - train_signal_only_llr.py (4 workers)
  - train_signal_only_llr_patd.py (8 workers)

- **Memory-constrained environments**: When running multiple processes on limited GPU/CPU

- **Reproducibility**: When exact random seed control is important

## Related scripts
- [train_chargenet.py](example-train_chargenet.md): Uses 4 DataLoader workers
- [train_signal_only_llr.py](example-train_signal_only_llr.md): Uses 4 DataLoader workers
- [train_signal_only_llr_patd.py](example-train_signal_only_llr_patd.md): Uses 8 DataLoader workers
- [res_test.py](example-res_test.md): Sets similar env vars inline at line 4

## See also
- PyTorch DataLoader documentation: https://pytorch.org/docs/stable/data.html
- OpenMP environment variables: https://www.openmp.org/spec-html/5.0/openmpsu59.html
- MKL documentation: https://www.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/environment-variables.html
