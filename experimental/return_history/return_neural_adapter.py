"""Versioned neural input adapter. No names/labels, optimizer or checkpoint I/O."""
import numpy as np
import torch
from torch import nn
from semantic_encoder_v3 import prepare_v3,SemanticEncoderV3

SCHEMA='bsfx-return-history-neural-input-r1'
TABLES=dict(calls=(128,7),returns=(128,7),guards=(128,7),summary_edges=(512,4),value_edges=(128,3),guard_edges=(256,10),continuation_edges=(512,5))
def require(ok,why):
 if not bool(ok):raise ValueError(why)
def raw_input(rep):
 require(rep.get('schema')=='bsfx-call-semantics-v3-return-repair-r1','REPRESENTATION_SCHEMA')
 require(rep.get('tensors') is not None,'CORE_UNAVAILABLE')
 relations=rep.get('return_flow',{});require(relations.get('tensors') is not None,'RETURN_RELATIONS_UNAVAILABLE')
 raw=dict(schema=SCHEMA,status=rep['status'],core={k:np.asarray(v,dtype=np.int64) for k,v in rep['tensors'].items()},
  node_mask=np.asarray(rep['legacy']['base']['tensors']['node_mask'],dtype=np.int64),tables={},masks={},history=np.zeros((2048,2),np.int64),history_mask=np.zeros(2048,np.int64))
 for key,(rows,width) in TABLES.items():
  values=relations['tensors'][key];require(len(values)<=rows,'TABLE_CAP:'+key)
  t=np.zeros((rows,width),np.int64);mask=np.zeros(rows,np.int64)
  if values:t[:len(values)]=np.asarray(values,np.int64);mask[:len(values)]=1
  raw['tables'][key]=t;raw['masks'][key]=mask
 history=rep['historical_guard_edges'];require(len(history)<=2048,'HISTORY_CAP')
 if history:raw['history'][:len(history)]=np.asarray(history,np.int64);raw['history_mask'][:len(history)]=1
 return raw
def prepare_return_history(raw):
 require(set(raw)=={'schema','status','core','node_mask','tables','masks','history','history_mask'},'INPUT_KEYS')
 require(raw['schema']==SCHEMA,'SCHEMA')
 require(set(raw['tables'])==set(TABLES) and set(raw['masks'])==set(TABLES),'TABLE_KEYS')
 def array(x,shape,name):require(isinstance(x,np.ndarray) and x.dtype==np.int64 and x.shape==shape,'SHAPE_OR_DTYPE:'+name)
 def mask(x,shape,name):
  array(x,shape,name);require(((x==0)|(x==1)).all(),'MASK:'+name);return np.flatnonzero(x)
 nodes=mask(raw['node_mask'],(12,),'nodes');require(0 in nodes,'FOCAL_NODE');nodes=set(map(int,nodes))
 core=prepare_v3(raw['core'],raw['status'])
 for e in core['events']:
  require(int(e[1]) in nodes and (int(e[4])==-1 or int(e[4]) in nodes),'CORE_INACTIVE_OWNER')
 for p in core['predicates']:require(int(p[1]) in nodes,'CORE_INACTIVE_OWNER')
 # Core adapter compacts active expression/predicate/path pools. Reconstruct maps.
 pm={int(v):i for i,v in enumerate(np.flatnonzero(raw['core']['predicate_mask']))}
 dm={int(v):i for i,v in enumerate(np.flatnonzero(raw['core']['path_mask']))}
 active={};tables={};maps={}
 for key,shape in TABLES.items():
  array(raw['tables'][key],shape,key);ids=mask(raw['masks'][key],(shape[0],),key)
  active[key]=ids;tables[key]=raw['tables'][key][ids].copy();maps[key]={int(old):i for i,old in enumerate(ids)}
 def ref(value,key):require(int(value) in maps[key],'INACTIVE_OR_BAD_REFERENCE:'+key);return maps[key][int(value)]
 def owner(value):require(int(value) in nodes,'INACTIVE_OWNER')
 def kind_slot(k,s):
  require(k in (0,1,2),'OBJECT_KIND');require(s==-1 if k==0 else 0<=s<128,'OBJECT_SLOT')
 calls=tables['calls'];returns=tables['returns'];guards=tables['guards']
 for c in calls:
  o,t,k,s,v,assigned,ordinal=map(int,c);owner(o);require(t==-1 or t in nodes,'CALLEE_OWNER');kind_slot(k,s)
  require(assigned in (0,1) and 0<=ordinal<512,'CALL_FLAGS')
  require((assigned==1 and k in (1,2) and 0<=v<128) or (assigned==0 and k==0 and v==-1),'CALL_DEFINITION')
 for r in returns:
  o,known,value,k,s,source,ordinal=map(int,r);owner(o);kind_slot(k,s);require(known in (0,1) and 0<=ordinal<512,'RETURN_FLAGS')
  if source>=0:
   compact=ref(source,'calls');require(calls[compact,0]==o,'RETURN_OWNER_MISMATCH');r[5]=compact
  else:require(source==-1,'RETURN_SOURCE')
  if not known:r[2]=0 # unknown active numeric payload is masked before float conversion
 for g in guards:
  o,k,s,v,truth,yes,no=map(int,g);owner(o);kind_slot(k,s);require(k in (1,2) and -1<=v<128,'GUARD_DEFINITION')
  require(truth in (0,1) and yes in (0,1) and no in (0,1),'GUARD_FLAGS')
 edge_sources=[];edge_targets=[];edge_types=[];edge_numeric=[]
 C=len(calls);N=C+len(returns)
 def edge(src,dst,kind,a,b):
  edge_sources.extend([src,dst]);edge_targets.extend([dst,src]);edge_types.extend([kind,kind+4]);edge_numeric.extend([[a,b],[a,b]])
 seen=set()
 for r in tables['summary_edges']:
  rid,cid,callee,may=map(int,r);rid=ref(rid,'returns');cid=ref(cid,'calls')
  require(may==1 and calls[cid,1]>=0 and returns[rid,0]==calls[cid,1]==callee,'SUMMARY_BINDING')
  key=('s',rid,cid);require(key not in seen,'DUPLICATE_EDGE');seen.add(key);edge(C+rid,cid,0,1,0)
 for r in tables['value_edges']:
  rid,cid,same=map(int,r);rid=ref(rid,'returns');cid=ref(cid,'calls')
  require(same==1 and returns[rid,0]==calls[cid,0] and returns[rid,5]==cid,'VALUE_BINDING')
  key=('v',rid,cid);require(key not in seen,'DUPLICATE_EDGE');seen.add(key);edge(cid,C+rid,1,1,0)
 linked={}
 for r in tables['guard_edges']:
  cid,gid,o,k,s,v,truth,yes,no,same=map(int,r);cid=ref(cid,'calls');gid=ref(gid,'guards');c=calls[cid];g=guards[gid]
  require(same==1 and c[0]==g[0]==o,'GUARD_OWNER')
  require(c[2]==g[1]==k and c[3]==g[2]==s and c[4]==g[3]==v,'GUARD_DEFINITION_MISMATCH')
  require(g[4]==truth and g[5]==yes and g[6]==no,'GUARD_EXIT_MISMATCH')
  key=('g',cid,gid);require(key not in seen,'DUPLICATE_EDGE');seen.add(key);linked[cid,gid]=(truth,yes,no);edge(cid,N+gid,2,truth,float(yes!=no))
 for r in tables['continuation_edges']:
  cid,gid,nid,truth,may=map(int,r);cid=ref(cid,'calls');gid=ref(gid,'guards');nid=ref(nid,'calls')
  require((cid,gid) in linked and may==1,'CONTINUATION_GUARD_MISSING');gtruth,yes,no=linked[cid,gid]
  require(yes!=no and truth==(1-gtruth if yes else gtruth),'CONTINUATION_POLARITY')
  require(calls[cid,0]==guards[gid,0]==calls[nid,0] and calls[nid,6]>calls[cid,6],'CONTINUATION_OWNER_ORDER')
  key=('c',cid,gid,nid);require(key not in seen,'DUPLICATE_EDGE');seen.add(key);edge(N+gid,nid,3,truth,1)
 array(raw['history'],(2048,2),'history');hi=mask(raw['history_mask'],(2048,),'history')
 history=[];history_seen=set();path_owners={p:set() for p in range(len(core['guards']))}
 for e,paths in zip(core['events'],core['event_paths']):
  for p in paths.tolist():path_owners[int(p)].add(int(e[1]))
 for p,q in raw['history'][hi]:
  require(int(p) in dm and int(q) in pm,'HISTORY_REFERENCE');p,q=dm[int(p)],pm[int(q)]
  require(all(int(core['predicates'][q,1])==o for o in path_owners[p]),'HISTORY_CROSS_OWNER')
  require(not any(int(g)==q and int(f)==1 for g,f in core['guards'][p]),'HISTORY_IS_FRESH_GUARD')
  require((p,q) not in history_seen,'HISTORY_DUPLICATE');history_seen.add((p,q));history.append([p,q])
 def tensor(v,shape,dtype=torch.long):return torch.tensor(v,dtype=dtype).reshape(shape)
 counts=[len(tables[k])/TABLES[k][0] for k in TABLES]+[len(history)/2048.]
 return dict(core=core,calls=tensor(calls,(-1,7)),returns=tensor(returns,(-1,7)),guards=tensor(guards,(-1,7)),
  edge_source=tensor(edge_sources,(-1,)),edge_target=tensor(edge_targets,(-1,)),edge_kind=tensor(edge_types,(-1,)),
  edge_numeric=tensor(edge_numeric,(-1,2),torch.float32),history=tensor(history,(-1,2)),counts=tensor(counts,(8,),torch.float32))

class ReturnHistoryEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.core=SemanticEncoderV3();self.object_kind=nn.Embedding(3,4);self.edge_kind=nn.Embedding(8,8)
  self.core.output.requires_grad_(False) # only core.encode is consumed; no unused trainable head
  self.call=nn.Sequential(nn.Linear(12,32),nn.Tanh());self.ret=nn.Sequential(nn.Linear(12,32),nn.Tanh());self.guard=nn.Sequential(nn.Linear(12,32),nn.Tanh())
  self.message=nn.Sequential(nn.Linear(74,32),nn.Tanh());self.update=nn.GRUCell(32,32);self.norm=nn.LayerNorm(32)
  self.history=nn.Sequential(nn.Linear(59,32),nn.Tanh());self.hidden=nn.Sequential(nn.Linear(200,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh())
  self.output=nn.Linear(32,2);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def encode(self,s,details=False):
  zero=self.output.weight.new_zeros;dev=self.output.weight.device
  require(s['calls'].device==dev and s['core']['events'].device==dev,'DEVICE_MISMATCH')
  core=self.core.encode(s['core'],details=True)
  def stacked(values,width):return torch.stack(values) if values else zero((0,width))
  cs=[];rs=[];gs=[];owners=[]
  for row in s['calls']:
   o,t,k,slot,v,known,ordinal=row;numeric=torch.stack([o/11.,(o==0).float(),t.clamp_min(0)/11.,(t>=0).float(),slot.clamp_min(0)/128.,v.clamp_min(0)/128.,known.float(),ordinal/511.])
   cs.append(self.call(torch.cat([numeric,self.object_kind(k)])));owners.append(o)
  for row in s['returns']:
   o,known,value,k,slot,source,ordinal=row;vf=value.float()
   numeric=torch.stack([o/11.,(o==0).float(),known.float(),torch.tanh(vf/16.),torch.sign(vf)*torch.log1p(vf.abs())/44.,slot.clamp_min(0)/128.,(source>=0).float(),ordinal/511.])
   rs.append(self.ret(torch.cat([numeric,self.object_kind(k)])));owners.append(o)
  for row in s['guards']:
   o,k,slot,v,truth,yes,no=row;numeric=torch.stack([o/11.,(o==0).float(),slot.clamp_min(0)/128.,v.clamp_min(0)/128.,(v>=0).float(),truth.float(),yes.float(),no.float()])
   gs.append(self.guard(torch.cat([numeric,self.object_kind(k)])));owners.append(o)
  h=torch.cat([stacked(cs,32),stacked(rs,32),stacked(gs,32)])
  source,target=s['edge_source'],s['edge_target']
  for _ in range(3):
   if len(source):
    msg=self.message(torch.cat([h[source],h[target],self.edge_kind(s['edge_kind']),s['edge_numeric']],-1))
    total=torch.zeros_like(h).index_add(0,target,msg);degree=h.new_zeros((len(h),1)).index_add(0,target,h.new_ones((len(target),1)))
    h=self.norm(self.update(total/degree.clamp_min(1),h))
  def pool(x,w):return torch.cat([x.mean(0),x.max(0).values]) if len(x) else zero(2*w)
  own=torch.stack(owners) if owners else torch.empty(0,dtype=torch.long,device=dev);focal=h[own==0]
  hp=[]
  for path,predicate in s['history']:
   expression,owner=s['core']['predicates'][predicate]
   numeric=torch.stack([owner/11.,(owner==0).float(),zero(())])
   hp.append(self.history(torch.cat([core['paths'][path],core['expressions'][expression],numeric])))
  hp=stacked(hp,32);features=torch.cat([core['encoding'][0],pool(h,32),focal.mean(0) if len(focal) else zero(32),pool(hp,32),s['counts']])
  require(features.shape==(200,),'FUSION_WIDTH');encoded=self.hidden(features).unsqueeze(0)
  return dict(encoding=encoded,node_encoding=h,history_encoding=hp,core_encoding=core['encoding'],feature_width=200) if details else encoded
 def forward(self,s):return self.output(self.encode(s))

class ReturnAugmentedModel(nn.Module):
 def __init__(self,legacy,new_branch=None,gate=0.):
  super().__init__();self.legacy=legacy;self.return_branch=new_branch if new_branch is not None else ReturnHistoryEncoder();self.gate=float(gate)
 def forward(self,batch,legacy_context,new_context=None):
  old=self.legacy(batch,legacy_context)
  delta=self.return_branch(new_context) if self.gate and new_context is not None else torch.zeros_like(old['logits'])
  require(delta.shape==old['logits'].shape,'BATCH_SHAPE_NO_BROADCAST')
  return dict(old,pre_return_logits=old['logits'],return_history_delta=delta,logits=old['logits']+self.gate*delta)
