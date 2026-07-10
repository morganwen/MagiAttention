# Magi_DSA V4 tests

These tests target one or two NVIDIA H100 80GB GPUs (SM90).  The frozen
development image is `magi-dsa-dev:v2`, local image id
`sha256:9c51e29d1fda8fc1a6e2a8e16c7b0773309e91dcbd44cf6d0182e5e3327c1029`.
Its relevant source pins are recorded in `docs/magi_dsa_v4_design.md`.

Compilation and execution are separate phases.  Build the image and external
kernels first; test commands must not silently compile a different revision.
The public-API reference suite can then be run with:

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/scratch.wewen_gpu:/ws \
  -w /ws/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll \
  magi-dsa-dev:v2 timeout 300 \
  pytest -q tests/test_dsa/test_dsa_api.py -m "not dsa_kernel"
```

Run the FlashMLA/cuDNN path explicitly in the frozen image:

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/scratch.wewen_gpu:/ws \
  -w /ws/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll \
  magi-dsa-dev:v2 timeout 300 \
  pytest -q tests/test_dsa/test_dsa_api.py -m dsa_kernel
```

Useful filters include `-k contract`, `-k compressor`, `-k reference`, and
`-k kernel`.  A compiled kernel execution over 10 seconds is treated as a
deadlock; later kernel unit tests use a 30-second watchdog and CP tests use a
60-second watchdog.  CP=2 data movement is intentionally not part of step 1.

The final directory responsibilities are:

- `test_dsa_api.py`: CP=1 public API, all layer forms, packed boundaries,
  output/KL/full-gradient parity, sink and compressor behavior.
- `test_dsa_cp.py`: CP=2 communication and full-path parity (steps 3-7).
- `test_dsa_dispatch.py`: fragment and transfer plans (step 2).
- `test_dsa_solver.py`: deterministic Indexer-balanced solver (step 2).
- `test_dsa_pack_kernel.py`: SM90 packing/remap/CSR kernels (step 4).
