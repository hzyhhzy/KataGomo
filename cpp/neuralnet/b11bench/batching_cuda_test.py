#!/usr/bin/env python3
"""Build a standalone REAL CUDA queue regression using an existing Linux Ninja build.

This only builds; it never starts CUDA, contacts a server, or rewrites katago.
First build the ordinary engine so its production objects are current. Then run:
  python3 batching_cuda_test.py /path/to/build
  /path/to/build/b11_batching_cuda_test MODEL CONFIG 13 3 exact 0 > result.jsonl
For every B13/B16, 1/3/12/13/16-row, exact/fullmask/partialmask case, omit selectors.
CONFIG should enable the normal B11 artifacts. Auto must select the real prepared
B11 lane; this test fails instead of silently comparing two unpadded evaluations.
Use an external timeout and run sequentially on the authorized single-GPU host.
"""
import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--ninja', default='ninja')
    args = parser.parse_args()
    build = args.build.resolve()
    source = Path(__file__).resolve().parent / 'batching_cuda_test.cpp'
    commands = subprocess.check_output(
        [args.ninja, '-C', str(build), '-t', 'commands', 'katago'],
        text=True, timeout=30).splitlines()
    if not any('-c ' in line and 'cudabackend.cpp.o' in line for line in commands):
        raise RuntimeError('The existing build must use the real CUDA backend')
    main_command = next(line for line in commands if '-c ' in line and '/main.cpp.o' in line)

    def compile_object(output, replacement=None, extra=()):
        command = shlex.split(main_command)
        if replacement is not None:
            command[command.index('-c') + 1] = str(replacement)
        command[command.index('-o') + 1] = str(output)
        if '-MF' in command:
            command[command.index('-MF') + 1] = str(output.with_suffix('.d'))
        if '-MT' in command:
            command[command.index('-MT') + 1] = str(output)
        command.extend(extra)
        subprocess.run(command, cwd=build, check=True, timeout=180)

    test_object = build / 'b11_batching_cuda_test.o'
    version_object = build / 'b11_batching_cuda_version.o'
    compile_object(test_object, source)
    # Keep Version helpers from main.cpp while replacing only the executable entry.
    compile_object(version_object, extra=['-Dmain=katago_original_main'])
    link_line = next(line for line in reversed(commands) if ' -o katago ' in line)
    command = [part for part in shlex.split(link_line) if part not in (':', '&&')]
    command = [part for part in command if not part.endswith('/main.cpp.o')]
    command[command.index('-o') + 1] = 'b11_batching_cuda_test'
    command[1:1] = [str(test_object), str(version_object)]
    # In particular, keep the normal cudabackend.cpp.o and all normal CUDA objects.
    subprocess.run(command, cwd=build, check=True, timeout=180)
    print(build / 'b11_batching_cuda_test')


if __name__ == '__main__':
    main()
