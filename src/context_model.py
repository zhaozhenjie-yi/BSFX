"""Learned directed call-context residual; zero-gate preserves the base model."""
import torch
from torch import nn

class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.types=nn.Embedding(22,8,padding_idx=0)
        self.resolution=nn.Embedding(3,4,padding_idx=0)
        self.node_input=nn.Sequential(nn.Linear(142,64),nn.ReLU(),nn.LayerNorm(64))
        self.edge_input=nn.Sequential(nn.Linear(76,64),nn.ReLU())
        self.message_in=nn.Sequential(nn.Linear(128,64),nn.ReLU())
        self.message_out=nn.Sequential(nn.Linear(128,64),nn.ReLU())
        self.update=nn.Linear(192,64);self.norm=nn.LayerNorm(64)
        self.hidden=nn.Sequential(nn.Linear(64,32),nn.ReLU())
        self.output=nn.Linear(32,2)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,c):
        # Single function per physical batch; node indices carry topology, not learned identities.
        features=c['node_features'];mask=c['node_mask'].float().unsqueeze(-1)
        numeric=features[:,[0,1,2,3,5]].float();numeric=numeric*numeric.new_tensor([1.,1.,1.,1/8.,1.])
        param=(self.types(c['fixed_parameter_types'])*c['fixed_parameter_mask'].unsqueeze(-1)).flatten(1)
        va=(self.types(c['va_arg_types'])*c['va_arg_mask'].unsqueeze(-1)).flatten(1)
        omitted=c['omitted_neighbour_edge_count'].float().unsqueeze(-1)/32.
        h=self.node_input(torch.cat([numeric,self.types(features[:,4]),param,va,omitted],dim=-1))*mask
        args=(self.types(c['call_argument_types'])*c['call_argument_mask'].unsqueeze(-1)).flatten(1)
        e=self.edge_input(torch.cat([args,self.resolution(c['call_resolution']),c['call_argument_mask'].float()],dim=-1))
        src=c['call_source'].clamp_min(0);dst=c['call_target'].clamp_min(0)
        valid=c['call_mask'].float().unsqueeze(-1)
        resolved=valid*c['call_target'].ge(0).unsqueeze(-1)
        # Only valid edges contribute. Unknown targets are explicitly zero, never aliased to focal node.
        for _ in range(2):
            incoming=self.message_in(torch.cat([h[src],e],-1))*resolved
            target_h=h[dst]*c['call_target'].ge(0).unsqueeze(-1)
            outgoing=self.message_out(torch.cat([target_h,e],-1))*valid
            mi=torch.zeros_like(h).index_add(0,dst,incoming)
            mo=torch.zeros_like(h).index_add(0,src,outgoing)
            di=torch.zeros_like(mask).index_add(0,dst,resolved).clamp_min(1.)
            do=torch.zeros_like(mask).index_add(0,src,valid).clamp_min(1.)
            h=self.norm(h+torch.relu(self.update(torch.cat([h,mi/di,mo/do],-1))))*mask
        assert int(features[0,0])==1
        return self.output(self.hidden(h[0])).unsqueeze(0)

class ContextAugmentedModel(nn.Module):
    def __init__(self,base,gate):
        super().__init__();self.base=base;self.context=ContextEncoder();self.gate=float(gate)
    def forward(self,batch,context):
        output=self.base(batch);delta=self.context(context)
        logits=output.final_logits+self.gate*delta
        return dict(logits=logits,base_logits=output.final_logits,base_unclipped_probability=output.final_probability,
                    context_delta=delta)
