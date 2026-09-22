#!/usr/bin/env bash
# Build the local Driver proxy; no Docker or system-library modifications.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for tool in cmake patchelf; do
    command -v "$tool" >/dev/null || { echo "Missing build dependency: $tool" >&2; exit 1; }
done
cmake -S "$ROOT/hijack" -B "$ROOT/hijack/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/hijack/build" --parallel "${LILT_BUILD_JOBS:-2}"
for role in hp be; do
    destination="$ROOT/hijack/$role-lib"
    mkdir -p "$destination"
    install -m 755 "$ROOT/hijack/build/liblilt-$role.so" "$destination/libcuda.so.1"
    patchelf --set-soname libcuda.so.1 "$destination/libcuda.so.1"
    ln -sfn libcuda.so.1 "$destination/libcuda.so"
    ln -sfn libcuda.so.1 "$destination/libcontroller.so"
done
echo "Lilt proxies: $ROOT/hijack/{hp,be}-lib/libcuda.so.1"
