from pathlib import Path
import os,sys,json,hashlib,random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent
E=Path('BSFX_WORKSPACE/experiments')
H=E/'automotive_context_v2_v3_full_eval_20261101_r1'
P=E/'school_data_validation_20261017_150757'
sys.path[:0]=[str(R),str(H),str(E/'context_v2_learning_ablation_20261024_r1'),str(E/'context_learning_ablation_20261024_r1'),str(E/'context_v3_encoder_20261025_r1'),str(P),str(P/'runtime_source')]
import numpy as np
import torch
from chapter4_repro.task34.model import Task34IntegratedModel,ENGINEERING_ALIGNED_V1
from chapter4_repro.task39s import r6e_execution_bridge as bridge
from semantic_encoder import ContextV2Model
from semantic_encoder_v3 import ContextV3Model
from missing_context_bridge import MissingAwareV2,MissingAwareV3
from evaluation_utils import tree_digest,digest,move
import common04 as historic

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def setup():
 torch.set_num_threads(4);torch.set_num_interop_threads(4)
 torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 assert torch.cuda.is_available()
 torch.cuda.set_per_process_memory_fraction(.75)
 free,total=torch.cuda.mem_get_info();assert free>4*1024**3,'INSUFFICIENT_FREE_GPU_MEMORY'
 torch.cuda.reset_peak_memory_stats()
 return dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),free_bytes=free,total_bytes=total,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),deterministic=True,memory_fraction=.75)
def model(arm):
 # Identical untrained historical backbone for all configurations; never load fitted ECU models.
 seed(123)
 base=Task34IntegratedModel(family_id=ENGINEERING_ALIGNED_V1,vocab_size=9279,padding_idx=0,transformer_layers=6,transformer_heads=8,d_model=512,dim_feedforward=3072,dropout=arm['dropout'],classifier_dropout=arm['dropout'],residual_branch_scale=.1)
 m=ContextV2Model(base,1.)
 cp=E/'context_v2_learning_ablation_20261024_r1/common_initial_state.pt'
 assert sha(cp)=='164f5b9d29aaa7022596454ac023e3bca523c71fd46f3adc4f807f7d91d26901'
 m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True)
 m.__class__=MissingAwareV2
 if arm['route']=='v3':
  torch.manual_seed(20261025);m=ContextV3Model(m,1.);m.__class__=MissingAwareV3
 return m
def metrics(pred,threshold=.5):
 y=np.array([r['label'] for r in pred],dtype=np.int64);p=np.array([r['probability'] for r in pred]);z=p>threshold
 tp=int(((y==1)&z).sum());tn=int(((y==0)&~z).sum());fp=int(((y==0)&z).sum());fn=int(((y==1)&~z).sum());pos=int(y.sum());neg=len(y)-pos
 f1p=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;f1n=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
 denom=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
 auc=float(sum((a>b)+.5*(a==b) for a in p[y==1] for b in p[y==0])/(pos*neg)) if pos*neg else None
 ap=0.;old=0
 for s in sorted(set(p.tolist()),reverse=True):
  selected=p>=s;new=int(y[selected].sum());ap+=(new-old)/pos*new/int(selected.sum()) if pos else 0.;old=new
 return dict(n=len(y),positive=pos,negative=neg,threshold=float(threshold),TP=tp,TN=tn,FP=fp,FN=fn,accuracy=(tp+tn)/len(y),precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/pos if pos else None,negative_recall=tn/neg if neg else None,F1=f1p,negative_F1=f1n,macro_F1=(f1p+f1n)/2,balanced_accuracy=(tp/pos+tn/neg)/2 if pos*neg else None,MCC=(tp*tn-fp*fn)/denom if denom else 0.,AUROC=auc,AP=float(ap),mean_CE=float(np.mean([r['cross_entropy'] for r in pred])))
def threshold_choice(pred):
 # Predefined finite grid; validation only. Never search test thresholds.
 options=[metrics(pred,t) for t in [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]]
 return max(options,key=lambda x:(x['macro_F1'],x['balanced_accuracy'],-abs(x['threshold']-.5)))
def load_data(split):
 rec=rows(R/'inputs'/f'{split}.jsonl')
 data={x['sample_id']:torch.load(R/'cache'/(x['sample_id']+'.pt'),map_location='cpu',weights_only=True) for x in rec}
 return rec,data
def predict(m,rec,data,arm,counts):
 m.eval();out=[]
 with torch.inference_mode():
  for r in rec:
   d=data[r['sample_id']];b=move(d['batch']);c=move(d['context']);counts['evaluation_forwards']+=1
   o=m(b,c if arm['route']=='v3' else c['v2']);z=o['logits'][0].double();assert torch.isfinite(z).all();p=float(z.softmax(-1)[1]);y=r['label']
   out.append(dict(sample_id=r['sample_id'],group_id=r['group_id'],file_path=r['file_path'],function_sha256=r['function_sha256'],label=y,probability=p,logits=z.cpu().tolist(),cross_entropy=float(torch.logsumexp(z,0)-z[y])))
 return out
