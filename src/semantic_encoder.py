"""Bounded semantic-event encoder. No labels or names are input features."""
import numpy as np
import torch
from torch import nn
from context_model import ContextAugmentedModel

def prepare_semantics(raw):
    """CPU-only adapter. Ignore masked values before indexing or decoding payloads."""
    legacy={k[4:]:torch.from_numpy(v.copy()).long() for k,v in raw.items() if k.startswith('v1__')}
    ids=np.flatnonzero(raw['event_mask']);emap={int(x):i for i,x in enumerate(ids)};n=len(ids)
    features=raw['event_features'][ids].copy();expr=[];guard_ids=np.full((n,4),-1,np.int64);arg_ids=np.full((n,8),-1,np.int64)
    for i,old in enumerate(ids):
        for maskkey,tokenskey,index in [('guard_mask','guard_tokens',guard_ids),('argument_mask','argument_tokens',arg_ids)]:
            for j in np.flatnonzero(raw[maskkey][old]):
                t=raw[tokenskey][old,j].copy();valid=t[:,0]!=0;length=int(valid.sum());assert length>0 and valid[:length].all() and not valid[length:].any()
                t=t[:length];assert ((t[:,0]>=0)&(t[:,0]<34)).all() and ((t[:,1]>=0)&(t[:,1]<22)).all()
                numeric=np.zeros((length,4),np.float32)
                for k,(op,typ,payload,arity) in enumerate(t):
                    if op in [4,5]:
                        val=float(payload) if op==4 else float(np.array([payload],dtype=np.int64).view(np.float64)[0]);assert np.isfinite(val)
                        numeric[k,0]=np.tanh(val/16.);numeric[k,1]=np.sign(val)*min(np.log1p(abs(val))/44.,1.)
                    elif op in [2,3,30]:numeric[k,2]=float(np.tanh(payload/16.))
                    numeric[k,3]=min(arity,8)/8.
                index[i,j]=len(expr);expr.append((t[:,:2].copy(),numeric))
    ne=len(expr);expr_ids=np.zeros((ne,32,2),np.int64);expr_num=np.zeros((ne,32,4),np.float32);lengths=np.zeros(ne,np.int64)
    for i,(x,y) in enumerate(expr):expr_ids[i,:len(x)]=x;expr_num[i,:len(x)]=y;lengths[i]=len(x)
    en=np.zeros((n,12),np.float32);prev=np.full(n,-1,np.int64)
    for i,e in enumerate(features):
        kind,owner,ordinal,target,ok,os,ck,cs,ct,cu,pr,dead=map(int,e)
        assert 1<=kind<=5 and 0<=owner<12 and ok in [0,1,2] and ck in [0,1,2]
        assert raw['v1__node_mask'][owner]==1
        prev[i]=emap.get(pr,-1)
        en[i]=[owner/11.,float(owner==0),min(ordinal,127)/127.,float(target>=0),max(target,0)/11.,float(target==0),max(os,0)/32.,max(cs,0)/32.,float(cu),float(pr>=0),float(dead),float(raw['function_partial'][owner])]
    b=raw['binding_features'][raw['binding_mask'].astype(bool)].copy();a=raw['alignment_features'][raw['alignment_mask'].astype(bool)].copy()
    bn=np.zeros((len(b),5),np.float32);an=np.zeros((len(a),7),np.float32)
    for i,x in enumerate(b):
        call,caller,kind,slot,callee,fkind,fslot,known=map(int,x);assert call in emap
        b[i,0]=emap[call];bn[i]=[caller/11.,max(slot,0)/32.,callee/11.,fslot/8.,float(known)]
    for i,x in enumerate(a):
        root,consumer,owner,kind,slot,pos,ct,at,known,present=map(int,x);assert root in emap and consumer in emap
        a[i,0]=emap[root];a[i,1]=emap[consumer];an[i]=[owner/11.,kind/2.,max(slot,0)/32.,max(pos,0)/8.,float(known),float(present),float(known and not present)]
    valid=raw['v1__node_mask'].astype(bool);partial=float(raw['function_partial'][valid].mean()) if valid.any() else 0.
    values=dict(expr_ids=expr_ids,expr_numeric=expr_num,expr_lengths=lengths,guard_indices=guard_ids,argument_indices=arg_ids,event_kind=features[:,0],event_object_kind=features[:,4],event_copy_kind=features[:,6],event_consumed_type=features[:,8],event_numeric=en,previous_event=prev,owner_is_focal=(features[:,1]==0),binding_call=b[:,0],binding_actual_kind=b[:,2],binding_formal_kind=b[:,5],binding_numeric=bn,alignment_root=a[:,0],alignment_consumer=a[:,1],alignment_consumed_type=a[:,6],alignment_actual_type=a[:,7],alignment_numeric=an,counts=np.array([n/128.,len(b)/64.,len(a)/64.,partial],np.float32))
    sem={k:torch.from_numpy(v.copy()) for k,v in values.items()}
    return dict(legacy=legacy,semantics=sem)

class SemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.op=nn.Embedding(34,8,padding_idx=0);self.types=nn.Embedding(22,4,padding_idx=0);self.position=nn.Embedding(32,4)
        self.expression=nn.GRU(20,24,batch_first=True)
        self.kind=nn.Embedding(6,8,padding_idx=0);self.object_kind=nn.Embedding(3,4)
        self.event=nn.Sequential(nn.Linear(320,64),nn.ReLU(),nn.LayerNorm(64))
        self.previous=nn.Sequential(nn.Linear(128,64),nn.ReLU());self.event_norm=nn.LayerNorm(64)
        self.binding=nn.Sequential(nn.Linear(77,32),nn.ReLU())
        self.alignment=nn.Sequential(nn.Linear(143,32),nn.ReLU())
        self.hidden=nn.Sequential(nn.Linear(260,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.output=nn.Linear(32,2);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def encode(self,s):
        device=self.output.weight.device;n=len(s['event_kind']);ne=len(s['expr_lengths'])
        if ne:
            token=s['expr_ids'];positions=torch.arange(32,device=device).expand(ne,-1)
            x=torch.cat([self.op(token[:,:,0]),self.types(token[:,:,1]),self.position(positions),s['expr_numeric']],-1)
            packed=nn.utils.rnn.pack_padded_sequence(x,s['expr_lengths'].cpu(),batch_first=True,enforce_sorted=False)
            _,h=self.expression(packed);eh=h[0]
        else:eh=self.output.weight.new_zeros((0,24))
        def gather_expr(ix):
            pad=torch.cat([eh,eh.new_zeros((1,24))]);return pad[torch.where(ix>=0,ix,torch.full_like(ix,ne))].flatten(1)
        if n:
            features=torch.cat([gather_expr(s['guard_indices']),gather_expr(s['argument_indices']),self.kind(s['event_kind']),self.object_kind(s['event_object_kind']),self.object_kind(s['event_copy_kind']),self.types(s['event_consumed_type']),s['event_numeric']],-1)
            h=self.event(features);prev=s['previous_event'];hp=h[prev.clamp_min(0)]*(prev>=0).unsqueeze(-1)
            h=self.event_norm(h+self.previous(torch.cat([h,hp],-1))*(prev>=0).unsqueeze(-1))
            focal=h[s['owner_is_focal']];pooled=torch.cat([h.mean(0),h.max(0).values,focal.mean(0) if len(focal) else h.new_zeros(64)])
        else:h=self.output.weight.new_zeros((0,64));pooled=self.output.weight.new_zeros(192)
        if len(s['binding_call']):
            b=self.binding(torch.cat([h[s['binding_call']],self.object_kind(s['binding_actual_kind']),self.object_kind(s['binding_formal_kind']),s['binding_numeric']],-1)).mean(0)
        else:b=pooled.new_zeros(32)
        if len(s['alignment_root']):
            a=self.alignment(torch.cat([h[s['alignment_root']],h[s['alignment_consumer']],self.types(s['alignment_consumed_type']),self.types(s['alignment_actual_type']),s['alignment_numeric']],-1)).mean(0)
        else:a=pooled.new_zeros(32)
        return self.hidden(torch.cat([pooled,b,a,s['counts']])).unsqueeze(0)
    def forward(self,s):return self.output(self.encode(s))

class ContextV2Model(ContextAugmentedModel):
    def __init__(self,base,semantic_gate):
        super().__init__(base,1);self.semantic=SemanticEncoder();self.semantic_gate=float(semantic_gate)
    def forward(self,batch,context):
        old=super().forward(batch,context['legacy']);new=self.semantic(context['semantics'])
        return dict(old,logits=old['logits']+self.semantic_gate*new,semantic_delta=new)
