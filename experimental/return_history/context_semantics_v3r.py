"""Versioned representation-only repair; not connected to historical encoders."""
import copy
import context_facts as v1
import context_semantics_v3 as legacy3
import context_semantics_v2r as v2r
from constant_values import IntegerConstants
import return_flow
SCHEMA='bsfx-call-semantics-v3-return-repair-r1'
class Extractor(legacy3.Extractor):
 def __init__(self,ast,focal,definitions,legacy):
  super().__init__(ast,focal,definitions,legacy);self.constants=IntegerConstants(ast)
 def expr(self,n):
  value=self.constants.value(n)
  if value is not None:return ['INT',v1.type_tag(v1.clean_type(n)),value,[]]
  return super().expr(n)
 def invalidate(self,states,reason):
  # Preserve historical predicates in an explicit sidecar, not as fresh guards.
  result=copy.deepcopy(states)
  for state in result:
   history=state.setdefault('historical_guards',[])
   for p,_ in state['guards']:
    if p not in history:history.append(p)
   state['guards']=[]
  if any(s['guards'] for s in states):self.note(reason)
  return result
 def facts(self):
  facts=super().facts();facts['constant_resolution']=self.constants.audit()
  facts['historical_path_guards']=[dict(path=i,predicates=p.get('historical_guards',[])) for i,p in enumerate(self.paths)]
  facts['guard_storage']='Fresh guards in core path tensors; invalidated evaluation-time guards retained in historical_guard_edges, not silently truncated.'
  return facts
def extract(ast,focal,definitions):
 legacy=v2r.extract(ast,focal,definitions);base=legacy['base']
 if base.get('tensors') is None:return dict(schema=SCHEMA,status='QUARANTINED_CONTEXT_CAP',reasons=base.get('reasons',[]),legacy=legacy,tensors=None,facts={})
 # Deliberately independent of V2 semantic-tensor success. Node scope is still bounded by V1.
 x=Extractor(ast,focal,definitions,legacy);t=None;reasons=[]
 try:
  t=x.run();history=[[i,p] for i,path in enumerate(x.paths) for p in path.get('historical_guards',[])]
  if len(history)>2048:raise legacy3.Cap('HISTORICAL_GUARD_EDGE_CAP')
 except legacy3.Cap as e:reasons=[str(e)];history=[];t=None
 names=[n['name'] for n in base['facts']['nodes']];relations=return_flow.build(ast,definitions,names)
 status='QUARANTINED_CAP' if t is None else 'MATERIALIZED_PARTIAL' if x.unknown else 'MATERIALIZED'
 return dict(schema=SCHEMA,status=status,reasons=reasons,legacy=legacy,tensors=t,facts=x.facts(),
  tensor_sha256=v1.digest(t) if t is not None else None,return_flow=relations,
  historical_guard_edges=history,combined_sha256=v1.digest(dict(core=t,history=history,return_flow=relations.get('tensors'))),
  limits=dict(legacy3.LIMITS),new_neural_encoder_connected=False,
  compatibility='Core shapes retained, guard meaning versioned and additional relation/history fields require an explicit future adapter; never feed silently to old checkpoints.')
