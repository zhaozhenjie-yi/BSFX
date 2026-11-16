from pathlib import Path
import subprocess,sys,json,os,time
R=Path(__file__).resolve().parent;start=time.time()
cmd=[sys.executable,'-B',str(R/'evaluate_all.py')]
with (R/'evaluation.stdout.log').open('xb') as o,(R/'evaluation.stderr.log').open('xb') as e:
    p=subprocess.run(cmd,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=o,stderr=e)
with (R/'EVALUATION_EXIT.json').open('x') as f:json.dump(dict(command=cmd,exit_code=p.returncode,seconds=time.time()-start,automatic_retry=False),f,indent=2)
sys.exit(p.returncode)
