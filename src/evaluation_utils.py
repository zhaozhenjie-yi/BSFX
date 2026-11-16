import json,hashlib,math
import torch
from sklearn.metrics import roc_auc_score,average_precision_score

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def digest(model):
 h=hashlib.sha256()
 for n,t in sorted(model.state_dict().items()):
  t=t.detach().cpu().contiguous();h.update(n.encode()+b'\0'+str(t.dtype).encode()+b'\0'+str(list(t.shape)).encode()+b'\0');h.update(t.numpy().tobytes())
 return h.hexdigest()

def tree_digest(value):
 h=hashlib.sha256()
 def can(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 def walk(x):
  if isinstance(x,torch.Tensor):
   t=x.detach().cpu().contiguous();h.update(can({'dtype':str(t.dtype),'shape':list(t.shape)}));h.update(t.numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x):h.update(str(k).encode()+b'\0');walk(x[k])
  elif isinstance(x,(tuple,list)):
   h.update(str(type(x)).encode());h.update(str(len(x)).encode())
   for y in x:walk(y)
  elif x is None or isinstance(x,(str,int,float,bool)):h.update(can(x))
  else:h.update(repr(x).encode())
 walk(value);return h.hexdigest()

def move(x):
 if isinstance(x,torch.Tensor):return x.to('cuda')
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 if isinstance(x,list):return [move(v) for v in x]
 if isinstance(x,tuple):return tuple(move(v) for v in x)
 return x

def metric(rows):
 y=[r['label'] for r in rows];p=[r['positive_probability'] for r in rows];pred=[r['predicted_label'] for r in rows]
 tp=sum(a==b==1 for a,b in zip(y,pred));tn=sum(a==b==0 for a,b in zip(y,pred));fp=sum(a==0 and b==1 for a,b in zip(y,pred));fn=sum(a==1 and b==0 for a,b in zip(y,pred))
 return dict(support=len(y),positive_support=sum(y),negative_support=len(y)-sum(y),TP=tp,TN=tn,FP=fp,FN=fn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),F1=2*tp/(2*tp+fp+fn),ROC_AUC=float(roc_auc_score(y,p)),Average_Precision=float(average_precision_score(y,p)))
