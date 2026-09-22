# Lilt

**Lilt: Transparent and Duration-Aware GPU Sharing for Inference–Training Colocation**

Lilt is a research GPU-sharing system for one high-priority (HP) inference
application and one or more best-effort (BE) training applications on one GPU.
Applications retain independent processes and CUDA contexts. No model, framework,
or GPU kernel modification is required; the task role is selected at launch.
The accompanying manuscript targets *Future Generation Computer Systems (FGCS)*.

Lilt combines event-guided HP activity tracking, duration-aware asynchronous BE
admission, and scheduling-granularity recovery for supported CUDA Graphs.
The HP activity monitor and completion tracker publish device completion state;
the Online Kernel Profiler and BE Admission Controller bound predicted in-flight
work while retaining CPU–GPU asynchrony. The graph-granularity recovery module
can divide supported static graphs into graphlets.

## Build

Requirements: Linux x86-64 with glibc, an NVIDIA driver, a C/C++ compiler,
CMake >= 3.13, Make, patchelf, and Python >= 3.10 for the launcher.

```bash
bash scripts/build_lilt.sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
cc -Wall -Wextra -Werror -Ihijack/include tests/test_lilt_config.c -o /tmp/lilt-config-test
/tmp/lilt-config-test
```

The build creates `hijack/hp-lib/libcuda.so.1` and
`hijack/be-lib/libcuda.so.1`. It does not modify system libraries or
`/etc/ld.so.preload`, start MPS, or change GPU compute mode.

## Launch

```bash
# Terminal 1: one HP process
python3 scripts/run_lilt.py hp --gpu 0 --session example01 -- \
    python3 examples/torch_workload.py hp --steps 1000 --batch-size 1

# Terminal 2: one BE process
python3 scripts/run_lilt.py be --gpu 0 --session example01 -- \
    python3 examples/torch_workload.py be --steps 1000 --batch-size 16
```
