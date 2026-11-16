"""Bounded, conservative source relations, not an error/safety classifier.

Each exact local/parameter declaration ID and assignment version is tracked.
Only direct call results and sibling scalar guards can form exit relations.
"""
import context_facts as v1
from constant_values import IntegerConstants
LIMITS=dict(calls=128,returns=128,guards=128,summary_edges=512,value_edges=128,guard_edges=256,continuation_edges=512)
def strip(n):
 while (n.get('kind') in ['ParenExpr','ConstantExpr'] or (n.get('kind')=='ImplicitCastExpr' and n.get('castKind') in ['NoOp','LValueToRValue','FunctionToPointerDecay','ArrayToPointerDecay'])) and len(n.get('inner',[]))==1:n=n['inner'][0]
 return n
def direct(n):
 n=strip(n);r=n.get('referencedDecl',{})
 return r.get('name') if n.get('kind')=='DeclRefExpr' and r.get('kind')=='FunctionDecl' else None
def build(ast,definitions,names):
 constants=IntegerConstants(ast);index={n:i for i,n in enumerate(names)};functions={f['name']:f for f in definitions};calls=[];returns=[];guards=[];ge=[];ve=[];ce=[];notes=[]
 for name in names:
  if name not in functions:continue
  owner=index[name];fn=functions[name];params=[n for n in fn.get('inner',[]) if n.get('kind')=='ParmVarDecl'];body=next(n for n in fn['inner'] if n.get('kind')=='CompoundStmt')
  locals_=[n for n in v1.walk(body) if n.get('kind')=='VarDecl'];symbols={n['id']:(1,i) for i,n in enumerate(params)};symbols.update({n['id']:(2,i) for i,n in enumerate(locals_)})
  versions={};ordinal=0
  def ref(n):
   n=strip(n);r=n.get('referencedDecl',{})
   return r.get('id') if n.get('kind')=='DeclRefExpr' and r.get('id') in symbols else None
  def guard(n):
   n=strip(n);r=ref(n)
   if r:return r,1,'NONZERO'
   cs=n.get('inner',[])
   if n.get('kind')=='UnaryOperator' and n.get('opcode')=='!' and len(cs)==1:
    r=ref(cs[0]);return (r,0,'ZERO') if r else None
   if n.get('kind')=='BinaryOperator' and n.get('opcode') in ['==','!='] and len(cs)==2:
    for a,b in [(cs[0],cs[1]),(cs[1],cs[0])]:
     r=ref(a)
     if r and constants.value(b)==0:return r,int(n['opcode']=='!='),'COMPARE_ZERO'
   return None
  def exits(n):
   if n.get('kind')=='ReturnStmt':return True
   if n.get('kind')=='CompoundStmt':
    cs=n.get('inner',[]);return len(cs)==1 and exits(cs[0])
   return False
  def scan(n,env,pending=()):
   nonlocal ordinal
   kind=n.get('kind');cs=n.get('inner',[])
   if kind=='CompoundStmt':
    state=dict(env);cont=list(pending)
    for child in cs:
     state,extra,alive=scan(child,state,cont);cont+=extra
     if not alive:return state,[],False
    return state,[],True
   if kind in ['ForStmt','WhileStmt','DoStmt','SwitchStmt','GotoStmt','IndirectGotoStmt','GCCAsmStmt']:
    notes.append(dict(owner=owner,reason='UNSUPPORTED_CONTROL_VALUE_FLOW_UNKNOWN',anchor=n.get('range',{})));return {},[],True
   if kind=='IfStmt':
    has_else=n.get('hasElse',False);g=guard(cs[0]) if cs else None;extra=[]
    if len(cs)!=(3 if has_else else 2):return {},[],True
    yes=exits(cs[1]);no=bool(has_else and exits(cs[2]));gid=len(guards)
    if g:
     rid,nonzero,form=g;definition=env.get(rid)
     guards.append(dict(id=gid,owner=owner,receiver=symbols[rid],version=calls[definition]['version'] if definition is not None else -1,true_when_nonzero=nonzero,
      direct_then_return=yes,direct_else_return=no,anchor=n.get('range',{})))
     if definition is not None:
      ge.append(dict(call_id=definition,guard_id=gid,owner=owner,receiver=symbols[rid],version=calls[definition]['version'],
       true_when_nonzero=nonzero,exit_on_true=yes,exit_on_false=no,same_definition=True))
      if yes!=no:extra.append(dict(call_id=definition,guard_id=gid,continue_when_nonzero=(1-nonzero) if yes else nonzero))
    scan(cs[1],dict(env),pending)
    if has_else:scan(cs[2],dict(env),pending)
    # No write merging across branches. Preserve only when branches cannot write.
    modified=any(x.get('kind') in ['CallExpr','VarDecl','CompoundAssignOperator'] or (x.get('kind') in ['BinaryOperator','UnaryOperator'] and x.get('opcode') in ['=','++','--']) for b in cs[1:] for x in v1.walk(b))
    return {} if modified else dict(env),extra,not(yes and no)
   if kind=='ReturnStmt':
    if any(x.get('kind')=='CallExpr' for x in v1.walk(n)):notes.append(dict(owner=owner,reason='RETURN_CONTAINS_CALL_RESULT_NOT_BOUND',anchor=n.get('range',{})))
    value=cs[0] if cs else {};known=constants.value(value);rid=ref(value);symbol=symbols.get(rid,(0,-1));retid=len(returns)
    returns.append(dict(id=retid,owner=owner,ordinal=ordinal,value_known=known is not None,value=known if known is not None else 0,
     object_kind=symbol[0],object_slot=symbol[1],source_definition_call=env.get(rid,-1),anchor=n.get('range',{})))
    if rid in env:ve.append(dict(return_id=retid,call_id=env[rid],same_definition=True))
    ordinal+=1;return dict(env),[],False
   receiver=None;value=None
   if kind=='DeclStmt' and len(cs)==1 and cs[0].get('kind')=='VarDecl':receiver=cs[0]['id'];value=cs[0].get('inner',[])[-1] if cs[0].get('inner') else None
   elif kind=='BinaryOperator' and n.get('opcode')=='=' and len(cs)==2:receiver=ref(cs[0]);value=cs[1]
   found=[x for x in v1.walk(n) if x.get('kind')=='CallExpr'];direct_call=strip(value) if value else None
   state=dict(env)
   for node in found:
    target=direct(node.get('inner',[{}])[0]);cid=len(calls);assigned=receiver is not None and direct_call is node
    if assigned:versions[receiver]=versions.get(receiver,-1)+1
    symbol=symbols.get(receiver,(0,-1)) if assigned else (0,-1)
    calls.append(dict(id=cid,owner=owner,ordinal=ordinal,callee=index.get(target,-1),callee_name=target,receiver=symbol,
     version=versions.get(receiver,-1) if assigned else -1,assignment_known=assigned,anchor=node.get('range',{})))
    for p in pending:ce.append(dict(**p,next_call_id=cid,may_reach=True))
    ordinal+=1
    # Other calls can escape or mutate the tracked state; do not infer alias safety.
    state={}
    if assigned:state[receiver]=cid
   if receiver is not None and not found:state.pop(receiver,None);versions[receiver]=versions.get(receiver,-1)+1
   if any(x.get('kind')=='UnaryOperator' and x.get('opcode') in ['++','--','&'] for x in v1.walk(n)):state={}
   return state,[],True
  scan(body,{})
 se=[dict(return_id=r['id'],call_id=c['id'],callee=c['callee'],may_return=True) for c in calls if c['callee']>=0 for r in returns if r['owner']==c['callee']]
 values=dict(calls=calls,returns=returns,guards=guards,summary_edges=se,value_edges=ve,guard_edges=ge,continuation_edges=ce)
 exceeded=[k for k,v in values.items() if len(v)>LIMITS[k]]
 facts=dict(**values,unknown_notes=notes,constant_resolution=constants.audit(),
  scope='Direct same-TU calls, exact scalar declaration/version, sibling guards and direct return exits; may-return edges do not prove path feasibility.',
  labels_used=False,no_general_alias_or_loop_proof=True)
 if exceeded:return dict(status='QUARANTINED_RELATION_CAP',reasons=exceeded,facts=facts,tensors=None)
 columns={
 'calls':lambda c:[c['owner'],c['callee'],*c['receiver'],c['version'],int(c['assignment_known']),c['ordinal']],
 'returns':lambda r:[r['owner'],int(r['value_known']),r['value'],r['object_kind'],r['object_slot'],r['source_definition_call'],r['ordinal']],
 'guards':lambda g:[g['owner'],*g['receiver'],g['version'],g['true_when_nonzero'],int(g['direct_then_return']),int(g['direct_else_return'])],
 'summary_edges':lambda e:[e['return_id'],e['call_id'],e['callee'],1],
 'value_edges':lambda e:[e['return_id'],e['call_id'],1],
 'guard_edges':lambda e:[e['call_id'],e['guard_id'],e['owner'],*e['receiver'],e['version'],e['true_when_nonzero'],int(e['exit_on_true']),int(e['exit_on_false']),1],
 'continuation_edges':lambda e:[e['call_id'],e['guard_id'],e['next_call_id'],e['continue_when_nonzero'],1]}
 tensors={k:[f(x) for x in values[k]] for k,f in columns.items()}
 return dict(schema='bsfx-return-result-guard-exit-r1',status='MATERIALIZED_PARTIAL' if notes else 'MATERIALIZED',facts=facts,tensors=tensors,
  tensor_sha256=v1.digest(tensors),neural_encoder_connected=False)
