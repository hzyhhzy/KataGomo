#!/usr/bin/env python3
"""Build and run the real NNEvaluator with a CPU-only fake backend.

No CUDA, model files, network access, or production-engine build is involved.
Requires a local C++ compiler, CMake, and zlib. Outputs stay in --build.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--generator')
    parser.add_argument('--zlib-include', type=Path)
    parser.add_argument('--zlib-library', type=Path)
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()
    if os.name == 'nt':
        # Inherited by child processes: a failed test must not open a modal WER
        # or critical-error dialog and hold an unattended run indefinitely.
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    here = Path(__file__).resolve().parent
    # Passing an explicit mapping also removes duplicate PATH/Path entries in
    # inherited Windows environment blocks, which otherwise break MSBuild.
    child_env = dict(os.environ)
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        command = [args.cmake, '-S', str(here / 'batching_cpu'), '-B', str(build)]
        generator = args.generator or ('Visual Studio 17 2022' if os.name == 'nt' else None)
        if generator:
            command += ['-G', generator]
            if generator.startswith('Visual Studio'):
                command += ['-A', 'x64']
        if args.zlib_include:
            command += ['-DZLIB_INCLUDE_DIR=' + str(args.zlib_include.resolve())]
        if args.zlib_library:
            command += ['-DZLIB_LIBRARY=' + str(args.zlib_library.resolve())]
        subprocess.run(command, check=True, timeout=60, env=child_env)
        subprocess.run([args.cmake, '--build', str(build), '--config', 'Release',
                        '--parallel', '1' if os.name == 'nt' else '8'],
                       check=True, timeout=240, env=child_env)
    runs = []
    for target in ('b11_batching_cpu_test', 'b11_batching_fake_cuda_test'):
        name = target + ('.exe' if os.name == 'nt' else '')
        executable = next((p for p in (build / 'Release' / name, build / name) if p.is_file()), None)
        if executable is None:
            raise RuntimeError('CPU test executable was not built: ' + target)
        result = subprocess.run([str(executable)], cwd=build, text=True, capture_output=True,
                                timeout=30, env=child_env)
        runs.append({'target': target, 'returncode': result.returncode,
                     'stdout': result.stdout, 'stderr': result.stderr})
        print(result.stdout, end='')
        print(result.stderr, end='', file=sys.stderr)
    report = {'runs': runs, 'gpu_execution': False, 'model_weights_loaded': False,
              'note': 'Fake CUDA build tests dispatch selection only; no CUDA runtime is linked.'}
    (build / 'batching_cpu_test.json').write_text(json.dumps(report, indent=2) + '\n')
    if any(run['returncode'] != 0 for run in runs):
        raise RuntimeError('One or more CPU batching tests failed; see batching_cpu_test.json')
    print(build / 'batching_cpu_test.json')


if __name__ == '__main__':
    main()
