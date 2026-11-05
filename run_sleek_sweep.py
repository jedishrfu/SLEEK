#!/usr/bin/env python3
"""Benchmark SLEEK binaries with an epsilon sweep and matched L-inf runs.

Example (run from the SLEEK directory after compiling standalone binaries):
    python3 run_sleek_sweep.py data.f32 -o results.csv
    python3 run_sleek_sweep.py data.f64 --precision double --backend gpu

Lossless ignores epsilon and has no -L2 option. It is repeated at each sweep
point for comparison. Exact reconstruction satisfies either norm.
Times include process startup and file I/O; MB/s uses decimal megabytes.
Only the Python standard library is required. Temporary codec files are removed.
Each -L2 run is followed by a run without -L2 using its measured L-inf
as the absolute error tolerance. Zero-error follow-ups are skipped because
SLEEK requires a positive lossy bound. source_epsilon links each follow-up
to the requested epsilon of its -L2 run. ARTIFACT builds receive the equivalent
range-relative tolerance.
Each row includes measured RMSE, L-inf, and unnormalized L2 errors against the
original input, separate from the requested epsilon. Binary floats use native
byte order, matching the standalone codecs. Nonfinite data/errors are rejected.
"""
import argparse
from array import array
import csv
import filecmp
import math
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
    'decompression_throughput_MB_s', 'mode', 'epsilon', 'rmse', 'l_inf', 'l2', 'source_epsilon',
]
MODES = [('lossless', False, False), ('lossy_with_L2', True, True)]
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


def reconstruction_errors(original, restored, precision):
    """Measure native binary floats in bounded chunks using double arithmetic.

    hypot avoids overflow/underflow from directly summing squared differences.
    L2 is the Euclidean norm; RMSE is L2 / sqrt(number of values).
    """
    typecode = 'f' if precision == 'single' else 'd'
    width = array(typecode).itemsize
    count = 0
    l_inf = l2 = 0.0
    with open(original, 'rb') as src, open(restored, 'rb') as dst:
        while True:
            raw = src.read(65536 * width)
            reconstructed = dst.read(65536 * width)
            if len(raw) != len(reconstructed) or len(raw) % width:
                raise RuntimeError('Invalid or mismatched reconstructed float data')
            if not raw:
                break
            values, decoded = array(typecode), array(typecode)
            values.frombytes(raw)
            decoded.frombytes(reconstructed)
            errors = []
            for a, b in zip(values, decoded):
                if not math.isfinite(a) or not math.isfinite(b):
                    raise RuntimeError('Cannot measure error for NaN or infinity')
                error = abs(a - b)
                if not math.isfinite(error):
                    raise RuntimeError('Reconstruction difference exceeds float64 range')
                errors.append(error)
            l_inf = max(l_inf, max(errors))
            l2 = math.hypot(l2, math.hypot(*errors))
            count += len(values)
    if not count or not math.isfinite(l2):
        raise RuntimeError('Empty input or L2 norm exceeds float64 range')
    return dict(rmse=l2 / math.sqrt(count), l_inf=l_inf, l2=l2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='Binary float32 or float64 file')
    parser.add_argument('-o', '--output', type=Path, default=Path('sleek_results.csv'))
    parser.add_argument('--append', action='store_true',
                        help='Append to an existing results CSV (header must match)')
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
    # ARTIFACT builds require the original input for lossy decompression.
    usage = subprocess.run([binaries['lossy', 'decompressor'], '--help'],
                           capture_output=True, text=True, timeout=args.timeout)
    needs_original = 'original_file_name' in usage.stdout + usage.stderr
    has_header = args.append and output.exists() and output.stat().st_size > 0
    if has_header:
        with output.open(newline='') as existing:
            if next(csv.reader(existing), None) != FIELDS:
                parser.error('Existing output CSV has an incompatible header')
    # Default exclusive creation protects reports; batch runs explicitly append.
    with output.open('a' if args.append else 'x', newline='') as report, tempfile.TemporaryDirectory(prefix='sleek-sweep-') as work:
        writer = csv.DictWriter(report, fieldnames=FIELDS)
        if not has_header:
            writer.writeheader()
        report.flush()
        compressed = Path(work) / 'compressed.bin'
        restored = Path(work) / 'restored.bin'
        print('Lossless rows ignore epsilon and L2 (exact reconstruction).', file=sys.stderr)
        row_count = 0

        def run_case(mode, lossy, l2, epsilon, source_epsilon=''):
            nonlocal row_count
            codec = 'lossy' if lossy else 'lossless'
            compressed.unlink(missing_ok=True)
            restored.unlink(missing_ok=True)
            comp = [binaries[codec, 'compressor'], str(src), str(compressed)]
            decomp = [binaries[codec, 'decompressor'], str(compressed), str(restored)]
            if lossy:
                comp.append(format(epsilon, '.17g'))
                decomp.append(format(epsilon, '.17g'))
                if needs_original:
                    decomp.append(str(src))
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
            errors = reconstruction_errors(src, restored, args.precision)
            writer.writerow(dict(
                file=src.name, command=f'SLEEK {args.precision} {args.backend} {codec}',
                command_args=shlex.join(comp[1:]) + ' ; decompress: ' + shlex.join(decomp),
                original_bytes=original_bytes, compressed_bytes=compressed_bytes,
                compression_ratio=original_bytes / compressed_bytes,
                exact_reconstruction=exact, compression_time_seconds=compression_time,
                compression_throughput_MB_s=original_bytes / 1e6 / compression_time,
                decompression_time_seconds=decompression_time,
                decompression_throughput_MB_s=original_bytes / 1e6 / decompression_time,
                mode=mode, epsilon=format(epsilon, '.17g'),
                source_epsilon=source_epsilon, **errors))
            report.flush()
            print(f'{mode}: epsilon={epsilon:.6g}, ratio={original_bytes / compressed_bytes:.4f}x',
                  file=sys.stderr)
            row_count += 1
            return errors

        input_range = None
        if needs_original:
            # ARTIFACT codecs interpret epsilon relative to the input range.
            minimum, maximum = math.inf, -math.inf
            with src.open('rb') as data:
                while raw := data.read(65536 * width):
                    values = array('f' if args.precision == 'single' else 'd')
                    values.frombytes(raw)
                    minimum = min(minimum, min(values))
                    maximum = max(maximum, max(values))
            input_range = maximum - minimum
        for mode, lossy, l2 in MODES:
            for epsilon in EPSILONS:
                errors = run_case(mode, lossy, l2, epsilon)
                if l2:
                    matched_epsilon = errors['l_inf']
                    if matched_epsilon == 0:
                        print(f'Skipping matched L-inf run for epsilon={epsilon:.6g}: '
                              'measured L-inf is zero; SLEEK requires a positive bound.',
                              file=sys.stderr)
                        continue
                    if input_range is not None:
                        if not math.isfinite(input_range) or input_range <= 0:
                            raise RuntimeError('Cannot convert L-inf to a range-relative tolerance')
                        matched_epsilon /= input_range
                    if not math.isfinite(matched_epsilon) or matched_epsilon <= 0:
                        raise RuntimeError('Matched L-inf epsilon is not positive and finite')
                    run_case('lossy_without_L2_matched_l_inf', True, False,
                             matched_epsilon, format(epsilon, '.17g'))
    print(f'Wrote {row_count} rows to {output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
