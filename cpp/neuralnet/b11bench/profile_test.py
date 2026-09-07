#!/usr/bin/env python3
"""Build/run pure CPU dispatch regression tests with existing Linux engine objects.

Example (no CUDA libraries linked and no GPU initialization):
  python3 cpp/neuralnet/b11bench/profile_test.py BUILD --model B11.bin.gz --run

BUILD must contain the matching engine's existing CPU object files. Outputs are
BUILD/b11_profile_test and BUILD/b11_profile_test.json, not engine objects. The
two small official Go transformer fixtures are read from cpp/tests/models by
default; --different-model can supply alternatives. All model hashes are checked
before/after execution. No checkpoint data is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path, help='Existing fully built Linux engine directory')
    parser.add_argument('--model', type=Path, help='Actual official tf3-b11c768 model; required with --run')
    parser.add_argument('--different-model', type=Path, action='append')
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run and args.model is None:
        parser.error('--model is required with --run')
    here = Path(__file__).resolve().parent
    cpp = here.parents[1]
    build = args.build.resolve()
    names = [
        'neuralnet/desc.cpp.o', 'neuralnet/modelversion.cpp.o',
        'core/global.cpp.o', 'core/fileutils.cpp.o', 'core/datetime.cpp.o',
        'core/sha2.cpp.o', 'core/rand.cpp.o', 'core/hash.cpp.o',
        'core/md5.cpp.o', 'core/timer.cpp.o', 'core/bsearch.cpp.o',
    ]
    objects = [build / 'CMakeFiles/katago.dir' / name for name in names]
    for path in objects:
        if not path.is_file():
            raise RuntimeError(f'missing existing CPU-only engine object: {path}')
    executable = build / 'b11_profile_test'
    command = [args.compiler, '-std=c++14', '-O2', '-Wall', '-Wextra',
               '-I' + str(cpp), str(here / 'profile_test.cpp'),
               *(str(path) for path in objects), '-lz', '-pthread', '-o', str(executable)]
    print(' '.join(command), flush=True)
    subprocess.run(command, check=True)
    if not args.run:
        return
    different = args.different_model or [
        cpp / 'tests/models/b7c96h3tfrs-test5-cnorm.bin.gz',
        cpp / 'tests/models/b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz',
    ]
    models = [args.model.resolve()] + [path.resolve() for path in different]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in models}
    run = subprocess.run([str(executable), *(str(path) for path in models)],
                         text=True, capture_output=True, timeout=120)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in models}
    result = {'command': command, 'models_sha256': before, 'models_unchanged': before == after,
              'returncode': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr,
              'gpu_execution': False}
    report = build / 'b11_profile_test.json'
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(run.stdout, end='')
    print(run.stderr, end='')
    if before != after:
        raise RuntimeError('a model file changed during CPU-only tests')
    run.check_returncode()
    print(report)


if __name__ == '__main__':
    main()
