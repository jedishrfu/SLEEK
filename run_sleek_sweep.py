#!/usr/bin/env python3
"""Benchmark standalone (non-ARTIFACT) SLEEK binaries with 30 round trips.

Example (run from the SLEEK directory after compiling standalone binaries):
    python3 run_sleek_sweep.py data.f32 -o results.csv
    python3 run_sleek_sweep.py data.f64 --precision double --backend gpu

Lossless ignores epsilon and has no -L2 option. It is repeated at each sweep
point for comparison. Exact reconstruction satisfies either norm.
Times include process startup and file I/O; MB/s uses decimal megabytes.
Only the Python standard library is required. Temporary codec files are removed.
"""
import argparse
import csv
import filecmp
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

FIELDS = [
    'file', 'command', 'command_args', 'original_bytes', 'compressed_bytes',
    'compression_ratio', 'exact_reconstruction', 'compression_time_seconds',
    'compression_throughput_MB_s', 'decompression_time_seconds',
    'decompression_throughput_MB_s', 'mode', 'epsilon',
]
MODES = [('lossless', False, False), ('lossy_without_L2', True, False),
         ('lossy_with_L2', True, True)]
EPSILONS = [0.01 * 1000 ** (i / 9) for i in range(10)]
EPSILONS[0], EPSILONS[-1] = 0.01, 10.0


def timed_run(command, timeout):
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {shlex.join(command)}\n'
                           f'{result.stdout}\n{result.stderr}')
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='Binary float32 or float64 file')
    parser.add_argument('-o', '--output', type=Path, default=Path('sleek_results.csv'))
    parser.add_argument('--precision', choices=['single', 'double'], default='single')
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--bin-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Directory containing standalone SLEEK executables')
    parser.add_argument('--timeout', type=float, default=300,
                        help='Maximum seconds per compressor/decompressor command (default: 300)')
    args = parser.parse_args()
    src = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not src.is_file():
        parser.error(f'Input file does not exist: {src}')
    original_bytes = src.stat().st_size
    width = 4 if args.precision == 'single' else 8
    if original_bytes == 0 or original_bytes % width:
        parser.error(f'Input must be nonempty and a multiple of {width} bytes')
    if args.timeout <= 0 or not args.timeout < float('inf'):
        parser.error('--timeout must be positive and finite')
    if output == src:
        parser.error('Output CSV must differ from the input file')
    suffix = '_cpu' if args.backend == 'cpu' else ''
    binaries = {}
    for codec in ['lossy', 'lossless']:
        for operation in ['compressor', 'decompressor']:
            path = (args.bin_dir.expanduser().resolve() /
                    f'sleek_{args.precision}_{operation}_{codec}{suffix}')
            if not path.is_file() or not os.access(path, os.X_OK):
                parser.error(f'Missing executable: {path}. Compile standalone SLEEK binaries first.')
            if output == path:
                parser.error('Output CSV must differ from codec executables')
            binaries[codec, operation] = str(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing reports from accidental replacement.
    with output.open('x', newline='') as report, tempfile.TemporaryDirectory(prefix='sleek-sweep-') as work:
        writer = csv.DictWriter(report, fieldnames=FIELDS)
        writer.writeheader()
        report.flush()
        compressed = Path(work) / 'compressed.bin'
        restored = Path(work) / 'restored.bin'
        print('Lossless rows ignore epsilon and L2 (exact reconstruction).', file=sys.stderr)
        for mode, lossy, l2 in MODES:
            codec = 'lossy' if lossy else 'lossless'
            for epsilon in EPSILONS:
                compressed.unlink(missing_ok=True)
                restored.unlink(missing_ok=True)
                comp = [binaries[codec, 'compressor'], str(src), str(compressed)]
                decomp = [binaries[codec, 'decompressor'], str(compressed), str(restored)]
                if lossy:
                    comp.append(format(epsilon, '.17g'))
                    decomp.append(format(epsilon, '.17g'))
                    if l2:
                        comp.append('-L2')
                        decomp.append('-L2')
                compression_time = timed_run(comp, args.timeout)
                compressed_bytes = compressed.stat().st_size
                if compressed_bytes <= 0:
                    raise RuntimeError(f'{mode}: compressor produced an empty file')
                decompression_time = timed_run(decomp, args.timeout)
                if restored.stat().st_size != original_bytes:
                    raise RuntimeError(f'{mode}: reconstructed size differs from input')
                exact = filecmp.cmp(src, restored, shallow=False)
                filecmp.clear_cache()
                if not lossy and not exact:
                    raise RuntimeError(f'{mode}: lossless reconstruction is not byte-identical')
                writer.writerow(dict(
                    file=src.name, command=f'SLEEK {args.precision} {args.backend} {codec}',
                    command_args=shlex.join(comp[1:]) + ' ; decompress: ' + shlex.join(decomp),
                    original_bytes=original_bytes, compressed_bytes=compressed_bytes,
                    compression_ratio=original_bytes / compressed_bytes,
                    exact_reconstruction=exact, compression_time_seconds=compression_time,
                    compression_throughput_MB_s=original_bytes / 1e6 / compression_time,
                    decompression_time_seconds=decompression_time,
                    decompression_throughput_MB_s=original_bytes / 1e6 / decompression_time,
                    mode=mode, epsilon=format(epsilon, '.17g')))
                report.flush()
                print(f'{mode}: epsilon={epsilon:.6g}, ratio={original_bytes / compressed_bytes:.4f}x',
                      file=sys.stderr)
    print(f'Wrote 30 rows to {output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
