"""CPU L2 round trips, legacy compatibility, and invalid bound regression checks."""
from pathlib import Path
import subprocess, struct, math, random, os, tempfile, shlex, atexit
r=Path(__file__).resolve().parents[1]
scratch=tempfile.TemporaryDirectory(prefix='sleek-l2-')
atexit.register(scratch.cleanup)
w=Path(scratch.name)
env=dict(os.environ, OMP_NUM_THREADS='2')
def run(args,ok=True):
 p=subprocess.run(list(map(str,args)),capture_output=True,env=env)
 assert (p.returncode==0)==ok, p.stdout.decode()+p.stderr.decode()
 return p
random.seed(27)
for precision,fmt in [('single','f'),('double','d')]:
 for artifact in [False,True]:
  bins=[]
  for op in ['compressor','decompressor']:
   exe=w/f'{precision}_{op}_{artifact}'
   args=shlex.split(os.environ.get('CXX', 'g++'))+['-O2','-fopenmp','-std=c++17','-I',r/'include',r/f'lossy_compressors_cpu/sleek_{precision}_{op}_lossy.cpp','-o',exe]
   if artifact: args+=['-DARTIFACT']
   run(args);bins.append(exe)
  for n in ([31,4097,20000] if artifact else [1,31,4097,20000]):
   vals=[random.uniform(-100,100) for _ in range(n)]
   if artifact: vals[0]=0; vals[-1]=100
   src=w/'input.bin';src.write_bytes(struct.pack(f'{n}{fmt}',*vals))
   vals=struct.unpack(f'{n}{fmt}',src.read_bytes())
   bound=0.1
   args_orig=[src] if artifact else []
   run([bins[0],src,w/'encoded',bound,'-L2'])
   run([bins[1],w/'encoded',w/'decoded',bound,*args_orig,'-L2'])
   dec=struct.unpack(f'{n}{fmt}',(w/'decoded').read_bytes())
   budget=bound*(max(vals)-min(vals)) if artifact else bound
   err=math.sqrt(math.fsum((a-b)**2 for a,b in zip(vals,dec)))
   assert err<=budget,(precision,artifact,n,err,budget)
  # Every position must produce identical compressed and reconstructed data.
  expected_encoded=(w/'encoded').read_bytes()
  expected_decoded=(w/'decoded').read_bytes()
  for pos in range(4):
   args=[src,w/'position_encoded',bound]
   args.insert(pos,'-L2')
   run([bins[0],*args])
   assert (w/'position_encoded').read_bytes()==expected_encoded
  for pos in range(4+len(args_orig)):
   args=[w/'encoded',w/'position_decoded',bound,*args_orig]
   args.insert(pos,'-L2')
   run([bins[1],*args])
   assert (w/'position_decoded').read_bytes()==expected_decoded
  run([bins[0],'-L2',src,w/'bad',bound,'-L2'],False)
  run([bins[0],src,w/'default',bound])
  run([bins[0],src,w/'bad',bound,'invalid'],False)
  run([bins[0],src,w/'bad',bound,'l2'],False)
  for b in ['0','-1','nan','inf','1e-38' if fmt=='f' else '1e-308']:
   run([bins[0],src,w/'bad',b,'-L2'],False)
  print(precision, 'artifact' if artifact else 'standalone', 'passed')
