#!/usr/bin/env python3
"""Build the CUDA-only diagnostic from an existing Linux Ninja engine build.

Reuses that build's exact compiler flags and engine objects. Does not run CUDA,
rewrite the engine executable, or require an installed Python package.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('build',type=Path,help='Existing configured and fully built Ninja build directory')
p.add_argument('--doom',action='store_true',help='Use the reference fork API adapter')
a=p.parse_args();build=a.build.resolve();source=Path(__file__).resolve().parent
commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands','katago'],text=True).splitlines()

def compile_object(command,output,replace_source=None,extra=()):
    args=shlex.split(command)
    if replace_source is not None:args[args.index('-c')+1]=str(replace_source)
    args[args.index('-o')+1]=str(output)
    # Manual diagnostics must not overwrite the engine object's dependency file.
    if '-MF' in args:args[args.index('-MF')+1]=str(output.with_suffix('.d'))
    if '-MT' in args:args[args.index('-MT')+1]=str(output)
    args.extend(extra)
    subprocess.run(args,cwd=build,check=True)

backend=next(x for x in commands if '-c ' in x and 'cudabackend.cpp.o' in x)
backend_source=Path(shlex.split(backend)[shlex.split(backend).index('-c')+1])
if not backend_source.is_absolute():backend_source=build/backend_source
cpp=backend_source.resolve().parents[1]
obj=build/'b11_inference_bench.o'
compile_object(backend,obj,source/'inference_bench.cpp',
               ['-I'+str(cpp)]+(['-DDOOM_REFERENCE'] if a.doom else []))
version=build/'b11_bench_version.o'
main=next(x for x in commands if '-c ' in x and '/main.cpp.o' in x)
compile_object(main,version,extra=['-Dmain=katago_original_main'])
link=next(x for x in reversed(commands) if ' -o katago ' in x)
args=[x for x in shlex.split(link) if x not in (':','&&')]
args=[x for x in args if not x.endswith('/main.cpp.o') and not x.endswith('/neuralnet/cudabackend.cpp.o')]
args[args.index('-o')+1]='b11_inference_bench'
args[1:1]=[str(obj),str(version)]
subprocess.run(args,cwd=build,check=True)
print(build/'b11_inference_bench')
