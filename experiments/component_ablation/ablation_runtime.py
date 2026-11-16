"""Isolated experiment adapters; historical modules and inputs remain read-only."""
from pathlib import Path
import sys, os, json, hashlib, importlib.util, types, dataclasses
R = Path(__file__).resolve().parent
PREV = Path('BSFX_WORKSPACE/experiments/xc2263n_report_training_20261110_r1')
spec = importlib.util.spec_from_file_location('previous_runtime', PREV/'runtime.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
torch, np = old.torch, old.np
read, rows, sha, save = old.read, old.rows, old.sha, old.save
seed, setup, move, metrics, predict = old.seed, old.setup, old.move, old.metrics, old.predict

def baseline_name(s):
    return 'v2_weighted_natural_800' if s == 123 else f'confirm_v2_weighted_natural_800_seed{s}'

def load_data(split):
    assert sha(R/'inputs'/f'{split}.jsonl') == sha(PREV/'inputs'/f'{split}.jsonl')
    return old.load_data(split)

def threshold_choice(pred):
    p=sorted(set(x['probability'] for x in pred))
    ts=sorted(set([0.,.5,1.]+[(a+b)/2 for a,b in zip(p,p[1:])]))
    return max((metrics(pred,t) for t in ts),key=lambda x:(x['macro_F1'],x['balanced_accuracy'],-abs(x['threshold']-.5)))

def freeze(module):
    for p in module.parameters(): p.requires_grad_(False)

def context_forward(self,batch,context):
    output=self.base(batch); zero=torch.zeros_like(output.final_logits)
    call=self.context(context['legacy']) if self._keep_call and context['availability']['legacy'] else zero
    sem=self.semantic(context['semantics']) if self._keep_sem and context['availability']['v2'] else zero
    return dict(logits=output.final_logits+self.gate*call+self.semantic_gate*sem,
                base_logits=output.final_logits,context_delta=call,semantic_delta=sem)

def semantic_only(self,token_ids,*args,padding_mask,**kwargs):
    value=self.semantic_projection(self.word_embedding(token_ids))
    return value.masked_fill(padding_mask.unsqueeze(-1),0.)

def identity_gru(self,x,*,padding_mask):
    valid=(~padding_mask).to(x.dtype)
    attention=valid/valid.sum(-1,keepdim=True).clamp_min(1)
    y=x.masked_fill(padding_mask.unsqueeze(-1),0.)
    return y,attention,(y*attention.unsqueeze(-1)).sum(1)

def identity_cnn(self,x,*,padding_mask):
    return x.masked_fill(padding_mask.unsqueeze(-1),0.),torch.ones_like(x[:,0])

def fixed_fusion(self,original,gru,cnn,*,padding_mask):
    context=self.output_projection((gru+cnn)*.5).masked_fill(padding_mask.unsqueeze(-1),0.)
    fused=self.concat_projection(torch.cat([original,context],-1)).masked_fill(padding_mask.unsqueeze(-1),0.)
    return fused,None

def model(arm):
    m=old.model(arm)
    initial=old.digest(m)
    a=arm['ablation']
    if a in ['no_context','no_call_context','no_semantic_context']:
        m._keep_call=a not in ['no_context','no_call_context']
        m._keep_sem=a not in ['no_context','no_semantic_context']
        if not m._keep_call: freeze(m.context)
        if not m._keep_sem: freeze(m.semantic)
        m.forward=types.MethodType(context_forward,m)
    elif a=='semantic_only_hpe':
        freeze(m.base.hpe)
        for mod in [m.base.hpe.word_embedding,m.base.hpe.semantic_projection]:
            for p in mod.parameters(): p.requires_grad_(True)
        m.base.hpe.forward=types.MethodType(semantic_only,m.base.hpe)
    elif a=='identity_gru':
        freeze(m.base.gru_branch);m.base.gru_branch.forward=types.MethodType(identity_gru,m.base.gru_branch)
    elif a=='identity_cnn':
        freeze(m.base.cnn_branch);m.base.cnn_branch.forward=types.MethodType(identity_cnn,m.base.cnn_branch)
    elif a=='fixed_dwaf_fusion':
        for mod in [m.base.dwaf.q_projection,m.base.dwaf.k_projection,m.base.dwaf.v_projection]:freeze(mod)
        m.base.dwaf.forward=types.MethodType(fixed_fusion,m.base.dwaf)
    elif a=='no_st_bias':
        for mod in m.base.transformer.modules():
            if type(mod).__name__=='StructureSensitiveSelfAttention':
                mod.config=dataclasses.replace(mod.config,structural_bias_enabled=False)
                freeze(mod.relation_bias)
    elif a=='uniform_segments':
        with torch.no_grad():m.base.classifier.probability_attention_weight.zero_()
        m.base.classifier.probability_attention_weight.requires_grad_(False)
        m.base.classifier.probability_attention_bias.requires_grad_(False)
    elif a=='alpha_one':
        for mod in m.base.transformer.modules():
            if type(mod).__name__=='StructureSensitiveTransformerLayer':mod.residual_branch_scale=1.
    elif a not in ['full','unweighted_ce']:raise ValueError(a)
    disabled=[n for n,p in m.named_parameters() if not p.requires_grad]
    info=dict(shared_initial_digest_before_intervention=initial,
              post_intervention_digest=old.digest(m),disabled_parameters=disabled,
              installed_parameters=sum(p.numel() for p in m.parameters()),
              trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
              frozen_parameters=sum(p.numel() for p in m.parameters() if not p.requires_grad))
    return m,info

def gradient_audit(m):
    result={}
    for n,p in m.named_parameters():
        k='.'.join(n.split('.')[:2]) if n.startswith('base.') else n.split('.')[0]
        v=result.setdefault(k,dict(parameters=0,trainable=0,grad_tensors=0,nonzero_grad_tensors=0,grad_norm_sq=0.))
        v['parameters']+=p.numel();v['trainable']+=p.numel()*int(p.requires_grad)
        if p.grad is not None:
            assert p.requires_grad,('disabled branch received gradient',n)
            v['grad_tensors']+=1;v['nonzero_grad_tensors']+=int(bool(p.grad.ne(0).any()))
            v['grad_norm_sq']+=float(p.grad.double().square().sum())
    return result

def schedule(s):
    p=PREV/'runs'/baseline_name(s)/'training_trace.jsonl'
    result=rows(p);assert len(result)==800
    assert sha(p)==read(R/'EXPERIMENT_CONFIG.json')['schedule_sha256'][str(s)]
    return result
