"""Versioned V3 residual adapter. No labels, sample identifiers or source names.

DNF aggregation represents structured alternatives; it is not a theorem prover.
Frozen V2 loads strictly into the separate `legacy` submodule.
"""
import numpy as np
import torch
from torch import nn

SHAPES = dict(event_mask=(256,),event_features=(256,6),event_path_mask=(256,16),event_paths=(256,16),expression_mask=(512,),expression_token_mask=(512,64),expression_tokens=(512,64,4),predicate_mask=(256,),predicate_features=(256,2),path_mask=(512,),path_unknown=(512,),path_guard_mask=(512,8),path_guards=(512,8,2),format_mask=(128,),format_features=(128,11))

def require(ok, message):
    if not bool(ok): raise ValueError(message)

def prepare_v3(raw, status):
    """Validate active data; ignore all masked payloads before any indexing.

    Pool indices are compacted, never embedded as sample-specific identities.
    Variable references in expressions are ordinal slots within the owner.
    """
    require(status in ('MATERIALIZED','MATERIALIZED_PARTIAL'), 'QUARANTINED_INPUT')
    require(isinstance(raw,dict) and set(raw)==set(SHAPES),'SCHEMA_KEYS')
    for k,shape in SHAPES.items():
        require(isinstance(raw[k],np.ndarray) and raw[k].shape==shape and raw[k].dtype==np.int64,'SHAPE_OR_DTYPE:'+k)
    def mask(k, row=None):
        v=raw[k] if row is None else raw[k][row]
        require(((v==0)|(v==1)).all(),'INVALID_MASK:'+k)
        return np.flatnonzero(v)
    ei=mask('event_mask');xi=mask('expression_mask');pi=mask('predicate_mask');di=mask('path_mask');fi=mask('format_mask')
    em={int(v):i for i,v in enumerate(ei)};xm={int(v):i for i,v in enumerate(xi)};pm={int(v):i for i,v in enumerate(pi)};dm={int(v):i for i,v in enumerate(di)}
    def ref(v,m,name):
        require(int(v) in m,'INVALID_REFERENCE:'+name);return m[int(v)]
    ids=np.zeros((len(xi),64,2),np.int64);nums=np.zeros((len(xi),64,5),np.float32);byte=np.zeros((len(xi),64),np.int64);lengths=[]
    for i,old in enumerate(xi):
        active=mask('expression_token_mask',old);require(len(active)>0,'EMPTY_EXPRESSION')
        require(np.array_equal(active,np.arange(len(active))),'NONPREFIX_EXPRESSION')
        tok=raw['expression_tokens'][old,active];slots=1
        for j,(op,typ,payload,arity) in enumerate(tok):
            op,typ,payload,arity=map(int,(op,typ,payload,arity))
            require(1<=op<40 and 0<=typ<22 and 0<=arity<=63 and slots>0,'INVALID_EXPRESSION')
            slots+=arity-1;ids[i,j]=[op,typ];num=nums[i,j];num[4]=arity/63.
            if op in (4,5):
                val=float(payload) if op==4 else float(np.array([payload],np.int64).view(np.float64)[0])
                require(np.isfinite(val),'NONFINITE_FLOAT')
                num[0]=np.tanh(val/16.);num[1]=np.sign(val)*min(np.log1p(abs(val))/710.,1.)
            elif op==36:
                require(0<=payload<=127 and arity==0,'INVALID_BYTE');byte[i,j]=payload+1
            elif op in (2,3,30,34,35,37,38):
                require(op not in (2,3,34,35) or payload>=0,'INVALID_SLOT')
                require(op!=30 or -1<=payload<12,'INVALID_CALLEE')
                require(op!=37 or 0<=payload<22,'INVALID_SIZEOF_TYPE')
                require(op!=38 or payload in (-2,-1,1,2),'INVALID_UPDATE')
                num[2]=np.tanh(payload/16.);num[3]=np.sign(payload)*min(np.log1p(abs(payload))/44.,1.)
        require(slots==0,'UNBALANCED_PREFIX_TREE');lengths.append(len(active))
    predicates=[]
    for old in pi:
        owner,ex=map(int,raw['predicate_features'][old]);require(0<=owner<12,'INVALID_PREDICATE_OWNER')
        predicates.append((ref(ex,xm,'predicate_expression'),owner))
    guards=[];unknown=[]
    for old in di:
        u=int(raw['path_unknown'][old]);require(u in (0,1),'INVALID_UNKNOWN');unknown.append(u);gs=[]
        for j in mask('path_guard_mask',old):
            p,fresh=map(int,raw['path_guards'][old,j]);require(fresh in (0,1),'INVALID_FRESHNESS');gs.append((ref(p,pm,'path_predicate'),fresh))
        guards.append(gs)
    events=[];paths=[]
    for old in ei:
        kind,owner,ordinal,ex,callee,dead=map(int,raw['event_features'][old])
        require(1<=kind<=10 and 0<=owner<12 and 0<=ordinal<256 and -1<=callee<12 and dead in (0,1),'INVALID_EVENT')
        ps=[ref(v,dm,'event_path') for v in raw['event_paths'][old,mask('event_path_mask',old)]]
        require(bool(ps)==(dead==0),'PATH_REACHABILITY_CONFLICT')
        for p in ps:
            require(all(predicates[g][1]==owner for g,_ in guards[p]),'CROSS_OWNER_GUARD')
        events.append((kind,owner,ordinal,ref(ex,xm,'event_expression'),callee,dead));paths.append(ps)
    formats=[]
    for old in fi:
        e,fam,conv,length,role,arg,expected,actual,present,known,relation=map(int,raw['format_features'][old])
        ec=ref(e,em,'format_event');require(events[ec][0]==1,'FORMAT_NONCALL')
        require(1<=fam<=4 and 0<=conv<=19 and 0<=length<=8 and 1<=role<=6 and -1<=arg<=255 and 0<=expected<=18 and 0<=actual<=18 and present in (0,1) and known in (0,1) and 0<=relation<=3,'INVALID_FORMAT')
        formats.append((ec,fam,conv,length,role,arg,expected,actual,present,known,relation))
    def tensor(x,dtype=torch.long):return torch.tensor(x,dtype=dtype)
    return dict(expr_ids=torch.from_numpy(ids),expr_numeric=torch.from_numpy(nums),expr_bytes=torch.from_numpy(byte),expr_lengths=tensor(lengths),predicates=tensor(predicates).reshape(-1,2),guards=[tensor(x).reshape(-1,2) for x in guards],path_unknown=tensor(unknown),events=tensor(events).reshape(-1,6),event_paths=[tensor(x) for x in paths],formats=tensor(formats).reshape(-1,11))

class SemanticEncoderV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.op=nn.Embedding(40,8,padding_idx=0);self.types=nn.Embedding(22,4,padding_idx=0);self.byte=nn.Embedding(129,4,padding_idx=0);self.position=nn.Embedding(64,4)
        self.expression=nn.GRU(25,24,batch_first=True)
        self.freshness=nn.Embedding(2,4);self.unknown=nn.Embedding(2,4)
        self.guard=nn.Sequential(nn.Linear(30,32),nn.Tanh())
        self.path=nn.Sequential(nn.Linear(69,32),nn.Tanh())
        self.kind=nn.Embedding(11,8,padding_idx=0)
        self.event=nn.Sequential(nn.Linear(103,64),nn.Tanh(),nn.LayerNorm(64))
        self.family=nn.Embedding(5,4);self.conversion=nn.Embedding(20,4);self.length=nn.Embedding(9,3);self.role=nn.Embedding(7,4);self.format_type=nn.Embedding(19,4);self.relation=nn.Embedding(4,3)
        self.format=nn.Sequential(nn.Linear(93,32),nn.Tanh())
        self.hidden=nn.Sequential(nn.Linear(259,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh())
        self.output=nn.Linear(32,2);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def encode(self,s,details=False):
        zero=self.output.weight.new_zeros;device=self.output.weight.device;n=len(s['expr_lengths'])
        if n:
            pos=torch.arange(64,device=device).expand(n,-1)
            x=torch.cat([self.op(s['expr_ids'][:,:,0]),self.types(s['expr_ids'][:,:,1]),self.byte(s['expr_bytes']),self.position(pos),s['expr_numeric']],-1)
            packed=nn.utils.rnn.pack_padded_sequence(x,s['expr_lengths'].cpu(),batch_first=True,enforce_sorted=False)
            _,h=self.expression(packed);eh=h[0]
        else:eh=zero((0,24))
        def pool(x,width):return torch.cat([x.mean(0),x.max(0).values]) if len(x) else zero(2*width)
        ph=[]
        for i,gs in enumerate(s['guards']):
            if len(gs):
                pred=s['predicates'][gs[:,0]];owner=pred[:,1]
                gh=self.guard(torch.cat([eh[pred[:,0]],owner.float().unsqueeze(-1)/11.,(owner==0).float().unsqueeze(-1),self.freshness(gs[:,1])],-1))
            else:gh=zero((0,32))
            ph.append(self.path(torch.cat([pool(gh,32),zero(1)+len(gs)/8.,self.unknown(s['path_unknown'][i])],-1)))
        ph=torch.stack(ph) if ph else zero((0,32));ev=[];live=[];focal=[]
        for i,row in enumerate(s['events']):
            kind,owner,ordinal,ex,callee,dead=row;ps=s['event_paths'][i]
            # Dead events retain auditable facts but contribute neither event nor format pooling.
            if int(dead):ev.append(zero(64));continue
            numeric=torch.stack([owner.float()/11.,(owner==0).float(),ordinal.float()/255.,(callee>=0).float(),callee.clamp_min(0).float()/11.,(callee==0).float(),zero(())+len(ps)/16.])
            ev.append(self.event(torch.cat([eh[ex],pool(ph[ps],32),self.kind(kind),numeric],-1)));live.append(i)
            if int(owner)==0:focal.append(i)
        ev=torch.stack(ev) if ev else zero((0,64));fs=[]
        for row in s['formats']:
            e,fam,conv,length,role,arg,expected,actual,present,known,relation=row
            if int(s['events'][e,5]):continue
            nums=torch.stack([arg.float()/255.,present.float(),known.float()])
            fs.append(self.format(torch.cat([ev[e],self.family(fam),self.conversion(conv),self.length(length),self.role(role),self.format_type(expected),self.format_type(actual),self.relation(relation),nums],-1)))
        fs=torch.stack(fs) if fs else zero((0,32))
        features=torch.cat([pool(ev[live],64),ev[focal].mean(0) if focal else zero(64),pool(fs,32),zero(3)+self.output.weight.new_tensor([len(live)/256.,len(fs)/128.,len(focal)/256.])])
        out=self.hidden(features).unsqueeze(0)
        return dict(encoding=out,events=ev,paths=ph,formats=fs,expressions=eh) if details else out
    def forward(self,s):return self.output(self.encode(s))

class ContextV3Model(nn.Module):
    """Keep V2 intact. Gate zero bypasses V3; zero new output head preserves logits."""
    def __init__(self,legacy,gate=1.):
        super().__init__();self.legacy=legacy;self.v3=SemanticEncoderV3();self.gate=float(gate)
    def forward(self,batch,context):
        old=self.legacy(batch,context['v2'])
        delta=self.v3(context['v3']) if self.gate else torch.zeros_like(old['logits'])
        return dict(old,legacy_logits=old['logits'],logits=old['logits']+self.gate*delta,v3_delta=delta)
