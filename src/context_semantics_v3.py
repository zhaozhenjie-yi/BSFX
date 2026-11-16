"""Additive, bounded C source representation. No model, labels or safety verdicts.

Legacy V2 tensors remain byte-for-byte compatible. New structured-flow facts
describe syntactic paths, not executable witnesses or general CFG proofs.
"""
import copy,json,re,ast as pyast,struct
import context_semantics_v2 as old

SCHEMA='bsfx-call-semantics-v3-format-structured-flow-r1'
LIMITS=dict(events=256,expressions=512,expression_tokens=64,predicates=256,
            paths=512,guards=8,alternatives=16,formats=128,format_bytes=512)
OPS=list(old.OP)+['GLOBAL','STRING','BYTE','SIZEOF','UPDATE','EMPTY']
OP={s:i for i,s in enumerate(OPS)}
KINDS={s:i+1 for i,s in enumerate(['CALL','RETURN','INIT','WRITE','INDEX_ACCESS','DIV','MOD','BRANCH','SHORT_CIRCUIT','UNKNOWN_CONTROL'])}
FTYPES=['UNKNOWN','int','unsigned int','long','unsigned long','long long','unsigned long long',
        'double','long double','char*','void*','int*','signed char*','short*','long*','long long*','unsigned long*','ARRAY','OTHER_POINTER']
FT={s:i for i,s in enumerate(FTYPES)}
FAMILY={'printf':(1,0,1),'fprintf':(2,1,2),'sprintf':(3,1,2),'snprintf':(4,2,3)}
CONV={s:i+1 for i,s in enumerate('diouxXfFeEgGaAcspn%')}
LENGTH={s:i for i,s in enumerate(['','hh','h','l','ll','j','z','t','L'])}

class Cap(Exception):pass
def key(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def clone(x):return copy.deepcopy(x)
def strip(n):return old.strip(n)
def children(n):return [x for x in n.get('inner',[]) if x.get('kind')]
def clean(n):return old.v1.clean_type(n)
def expression_tokens(e):
 op,typ,payload,args=e
 return [[OP[op],typ,payload,len(args)]]+[t for a in args for t in expression_tokens(a)]
def neg(e):
 if e[0]=='NOT':return clone(e[3][0])
 if e[0]=='INT':return ['INT',e[1],int(not e[2]),[]]
 if e[0] in ['LT','LE']:return ['LE' if e[0]=='LT' else 'LT',e[1],0,[clone(e[3][1]),clone(e[3][0])]]
 if e[0] in ['EQ','NE']:return ['NE' if e[0]=='EQ' else 'EQ',e[1],0,clone(e[3])]
 return ['NOT',e[1],0,[clone(e)]]
def decode_string(n):
 n=strip(n)
 if n.get('kind')!='StringLiteral':return None,'NON_LITERAL_FORMAT'
 if n.get('kind')=='StringLiteral' and n.get('type',{}).get('qualType','').startswith(('wchar','char16','char32')):return None,'WIDE_STRING_UNSUPPORTED'
 value=n.get('value','')
 if not value.startswith('"'):return None,'NON_NARROW_LITERAL'
 try:value=pyast.literal_eval(value)
 except (ValueError,SyntaxError):return None,'UNSUPPORTED_C_STRING_ESCAPE'
 if not isinstance(value,str):return None,'UNSUPPORTED_C_STRING'
 # Narrow ASCII fixtures are exact. Non-ASCII execution encodings are explicit unknown.
 if any(ord(c)>127 for c in value):return None,'EXECUTION_CHARSET_UNPROVEN'
 return value.split('\0',1)[0],None
def actual_format_type(n):
 t=re.sub(r'\s*\*\s*','*',clean(n))
 aliases={'signed int':'int','long int':'long','unsigned long int':'unsigned long',
          'long long int':'long long','unsigned long long int':'unsigned long long',
          'long int*':'long*','short int*':'short*','long long int*':'long long*','unsigned long int*':'unsigned long*'}
 t=aliases.get(t,t)
 if t in FT:return t
 if '*' in t:return 'OTHER_POINTER'
 if '[' in t:return 'ARRAY'
 return 'UNKNOWN'
def parse_format(text):
 """C printf subset, x86_64 LP64. Positionals/wide/unsupported forms stay unknown."""
 if len(text.encode('ascii'))>LIMITS['format_bytes']:raise Cap('FORMAT_BYTE_CAP')
 specs=[];unknown=[];i=0;arg=0
 while i<len(text):
  if text[i]!='%':i+=1;continue
  start=i;i+=1
  if i<len(text) and text[i]=='%':specs.append(dict(conv='%',length='',role='escaped',slot=-1,expected='UNKNOWN',offset=start));i+=1;continue
  tail=text[i:]
  if re.match(r'\d+\$',tail):unknown.append('POSITIONAL_FORMAT_UNSUPPORTED');break
  while i<len(text) and text[i] in '-+ #0':i+=1
  for role in ['width','precision']:
   if role=='precision':
    if i>=len(text) or text[i]!='.':continue
    i+=1
   if i<len(text) and text[i]=='*':
    specs.append(dict(conv='*',length='',role=role,slot=arg,expected='int',offset=start));arg+=1;i+=1
    if i<len(text) and (text[i].isdigit() or text[i]=='$'):unknown.append('POSITIONAL_STAR_UNSUPPORTED');return specs,unknown
   else:
    while i<len(text) and text[i].isdigit():i+=1
  length=''
  for candidate in ['hh','ll','h','l','j','z','t','L']:
   if text.startswith(candidate,i):length=candidate;i+=len(candidate);break
  if i>=len(text):unknown.append('INCOMPLETE_FORMAT');break
  conv=text[i];i+=1;expect='UNKNOWN'
  if conv in 'diouxX':
   unsigned=conv in 'ouxX'
   if length in ['h','hh']:expect='int' # LP64 int can represent all unsigned char/short values.
   elif length=='':expect='unsigned int' if unsigned else 'int'
   elif length=='l':expect='unsigned long' if unsigned else 'long'
   elif length=='ll':expect='unsigned long long' if unsigned else 'long long'
   elif length in ['j','z','t']:expect='unsigned long' if unsigned else 'long'
   else:unknown.append('INVALID_INTEGER_LENGTH')
  elif conv in 'fFeEgGaA':
   if length in ['','l']:expect='double'
   elif length=='L':expect='long double'
   else:unknown.append('INVALID_FLOAT_LENGTH')
  elif conv=='c' and not length:expect='int'
  elif conv=='s' and not length:expect='char*'
  elif conv=='p' and not length:expect='void*'
  elif conv=='n':expect={'':'int*','hh':'signed char*','h':'short*','l':'long*','ll':'long long*','j':'long*','z':'long*','t':'long*'}.get(length,'UNKNOWN')
  else:unknown.append('UNSUPPORTED_CONVERSION_OR_LENGTH')
  specs.append(dict(conv=conv,length=length,role='value',slot=arg,expected=expect,offset=start));arg+=1
 return specs,sorted(set(unknown))

class Extractor:
 def __init__(self,ast,focal,definitions,legacy):
  self.ast=ast;self.focal=focal;self.legacy=legacy
  self.names=[n['name'] for n in legacy['base']['facts']['nodes']];self.indices={n:i for i,n in enumerate(self.names)}
  self.funcs={f['name']:f for f in definitions};self.events=[];self.formats=[];self.predicates=[];self.unknown=set();self.function_unknown={}
  self.expressions=[];self.expression_map={};self.paths=[];self.path_map={};self.global_slots={}
 def note(self,reason):self.unknown.add(reason);self.un.add(reason)
 def intern_expr(self,e):
  ts=expression_tokens(e)
  if len(ts)>LIMITS['expression_tokens']:raise Cap('EXPRESSION_TOKEN_CAP')
  k=key(e)
  if k not in self.expression_map:
   if len(self.expressions)>=LIMITS['expressions']:raise Cap('EXPRESSION_POOL_CAP')
   self.expression_map[k]=len(self.expressions);self.expressions.append(e)
  return self.expression_map[k]
 def expr(self,n):
  n=strip(n);kind=n.get('kind');cs=children(n);typ=old.v1.type_tag(clean(n))
  def make(op,p=0,args=()):return [op,typ,p,list(args)]
  if kind=='DeclRefExpr':
   r=n.get('referencedDecl',{});symbol=self.symbols.get(r.get('id'))
   if symbol:return make(symbol[0],symbol[1])
   if r.get('kind')=='VarDecl':
    if r.get('id') not in self.global_slots:self.global_slots[r['id']]=len(self.global_slots)
    return make('GLOBAL',self.global_slots[r['id']])
   self.note('EXTERNAL_SYMBOL_VALUE_UNKNOWN');return make('UNKNOWN')
  if kind in ['IntegerLiteral','CharacterLiteral']:
   try:
    value=int(n['value'])
    if not -(2**63)<=value<2**63:raise ValueError()
    return make('INT',value)
   except (KeyError,ValueError):self.note('INTEGER_RANGE');return make('UNKNOWN')
  if kind=='FloatingLiteral':
   try:return make('FLOAT',struct.unpack('<q',struct.pack('<d',float(n['value'])))[0])
   except (ValueError,KeyError,OverflowError):self.note('FLOAT_UNKNOWN');return make('UNKNOWN')
  if kind=='StringLiteral':
   value,reason=decode_string(n)
   if reason:self.note(reason);return make('UNKNOWN')
   # Literal bytes retained only within this bounded syntax representation; no identifiers/labels.
   return make('STRING',len(value),[['BYTE',0,ord(c),[]] for c in value])
  if kind=='UnaryExprOrTypeTraitExpr' and n.get('name')=='sizeof':
   return make('SIZEOF',old.v1.type_tag(n.get('argType',{}).get('qualType','')),list(map(self.expr,cs)))
  if kind in ['CStyleCastExpr','ImplicitCastExpr'] and len(cs)==1:return make('CAST',args=[self.expr(cs[0])])
  if kind=='UnaryOperator' and len(cs)==1:
   op=n.get('opcode');a=self.expr(cs[0])
   if op in ['++','--']:return make('UPDATE',(1 if op=='++' else -1)*(2 if n.get('isPostfix') else 1),[a])
   if op in old.UN:
    tag=old.UN[op]
    if tag in ['NEG','POS'] and a[0]=='INT':return make('INT',-a[2] if tag=='NEG' else a[2])
    return make(tag,args=[a])
  if kind in ['BinaryOperator','CompoundAssignOperator'] and len(cs)==2:
   op=n.get('opcode');a,b=map(self.expr,cs)
   if kind=='CompoundAssignOperator':op=op[:-1];return make('ASSIGN',args=[a,make(old.BIN.get(op,'UNKNOWN'),args=[a,b])])
   if op in ['>','>=']:op,a,b=('<' if op=='>' else '<='),b,a
   if op=='+' and b[0]=='INT' and b[2]<0:b=clone(b);b[2]=-b[2];op='-'
   if op in old.BIN:return make(old.BIN[op],args=[a,b])
  if kind=='ArraySubscriptExpr' and len(cs)==2:return make('INDEX',args=list(map(self.expr,cs)))
  if kind=='CallExpr' and cs:
   ref=old.direct(cs[0]);return make('CALL',self.indices.get(ref.get('name'),-1) if ref else -1,list(map(self.expr,cs[1:])))
  if kind=='ConditionalOperator' and len(cs)==3:return make('COND',args=list(map(self.expr,cs)))
  if not kind:return make('EMPTY')
  self.note('EXPRESSION_'+str(kind));return make('UNKNOWN')
 def states_unique(self,states):
  unique={key(s):s for s in states}
  if len(unique)>LIMITS['alternatives']:raise Cap('PATH_ALTERNATIVE_CAP')
  return list(unique.values())
 def invalidate(self,states,reason):
  result=clone(states)
  for s in result:
   for g in s['guards']:g[1]=0
  if any(s['guards'] for s in states):self.note(reason)
  return result
 def branch(self,states,expr,truth,anchor):
  e=clone(expr) if truth else neg(expr)
  if e[0]=='INT':return clone(states) if e[2] else []
  exprid=self.intern_expr(e);pid=len(self.predicates)
  if pid>=LIMITS['predicates']:raise Cap('PREDICATE_CAP')
  self.predicates.append(dict(owner=self.owner,expression=exprid,condition=e,anchor=anchor))
  result=[]
  for state in states:
   state=clone(state)
   if len(state['guards'])>=LIMITS['guards']:raise Cap('GUARD_CAP')
   state['guards'].append([pid,1]);result.append(state)
  return result
 def intern_path(self,state):
  k=key(state)
  if k not in self.path_map:
   if len(self.paths)>=LIMITS['paths']:raise Cap('PATH_POOL_CAP')
   self.path_map[k]=len(self.paths);self.paths.append(clone(state))
  return self.path_map[k]
 def emit(self,kind,n,states,expr=None,callee=None):
  if len(self.events)>=LIMITS['events']:raise Cap('EVENT_CAP')
  states=self.states_unique(states);eid=len(self.events)
  item=dict(id=eid,kind=kind,owner=self.owner,owner_name=self.name,ordinal=self.ordinal,
    expression=self.intern_expr(expr if expr is not None else self.expr(n)),paths=[self.intern_path(s) for s in states],
    path_conditions=[dict(guards=[dict(predicate=p,condition=self.predicates[p]['condition'],current_value_valid=bool(f)) for p,f in s['guards']],control_unknown=s['unknown']) for s in states],
    syntactically_unreachable=not states,callee_name=callee,callee=self.indices.get(callee,-1),anchor=n.get('range',{}))
  self.events.append(item);self.ordinal+=1;return eid
 def format_call(self,node,eid,target,args):
  if target not in FAMILY:return
  if target in self.funcs:self.note('USER_DEFINED_PRINTF_NAME_NOT_ASSUMED_LIBC');return
  family,fmtidx,first=FAMILY[target]
  if len(args)<=fmtidx:self.note('FORMAT_ARGUMENT_ABSENT');return
  text,reason=decode_string(args[fmtidx])
  if reason:self.note(reason);specs=[];unknown=[reason]
  else:specs,unknown=parse_format(text)
  self.unknown.update(unknown);self.un.update(unknown)
  if not specs:
   specs=[dict(conv='',length='',role='unknown_format' if unknown else 'no_conversion',slot=-1,expected='UNKNOWN',offset=-1)]
  for spec in specs:
   if len(self.formats)>=LIMITS['formats']:raise Cap('FORMAT_RECORD_CAP')
   slot=spec['slot'];argidx=first+slot if slot>=0 else -1;present=0<=argidx<len(args);actual=actual_format_type(args[argidx]) if present else 'UNKNOWN'
   known=spec['expected']!='UNKNOWN' and not unknown
   # A type relation, never a vulnerability/safety verdict. Signedness exceptions
   # and pointed-object validity still require runtime values and full contracts.
   relation='UNKNOWN' if not known else 'MISSING_ARGUMENT' if not present else 'TYPE_EQUAL' if actual==spec['expected'] else 'TYPE_DIFF' if actual!='UNKNOWN' else 'UNKNOWN'
   self.formats.append(dict(call_event=eid,family=family,**spec,argument_index=argidx,actual=actual,present=present,known=known,relation=relation,format_unknown=unknown))
 def flow_expr(self,n,states):
  n=strip(n);kind=n.get('kind');cs=children(n)
  if kind=='BinaryOperator' and n.get('opcode') in ['&&','||'] and len(cs)==2:
   left=self.expr(cs[0]);s=self.flow_expr(cs[0],states);truth=n['opcode']=='&&'
   self.emit('SHORT_CIRCUIT',n,s)
   right=self.branch(s,left,truth,n.get('range',{}));other=self.branch(s,left,not truth,n.get('range',{}))
   right=self.flow_expr(cs[1],right)
   effects=any(x.get('kind')=='CallExpr' or x.get('kind')=='CompoundAssignOperator' or (x.get('kind') in ['BinaryOperator','UnaryOperator'] and x.get('opcode') in ['=','++','--']) for x in old.v1.walk(n))
   if not effects:return s
   return self.states_unique(other+right)
  if kind=='ConditionalOperator' and len(cs)==3:
   cond=self.expr(cs[0]);s=self.flow_expr(cs[0],states)
   a=self.flow_expr(cs[1],self.branch(s,cond,True,n.get('range',{})));b=self.flow_expr(cs[2],self.branch(s,cond,False,n.get('range',{})))
   return self.states_unique(a+b)
  s=clone(states)
  if kind=='CallExpr':
   if any(x.get('kind') in ['CallExpr','UnaryOperator'] and (x.get('kind')=='CallExpr' or x.get('opcode') in ['++','--']) for a in cs[1:] for x in old.v1.walk(a)):
    self.note('ARGUMENT_EVALUATION_ORDER_UNSPECIFIED')
    for q in s:q['unknown']=True
   for a in cs[1:]:s=self.flow_expr(a,s)
   ref=old.direct(cs[0]) if cs else None;target=ref.get('name') if ref else None
   if target is None:self.note('INDIRECT_CALL_TARGET_UNKNOWN')
   eid=self.emit('CALL',n,s,callee=target);self.format_call(n,eid,target,cs[1:])
   return self.invalidate(s,'CALL_EFFECT_MAY_INVALIDATE_CURRENT_GUARD_VALUES')
  for c in cs:s=self.flow_expr(c,s)
  if kind=='ArraySubscriptExpr':self.emit('INDEX_ACCESS',n,s)
  if kind=='BinaryOperator' and n.get('opcode') in ['/','%']:self.emit('DIV' if n['opcode']=='/' else 'MOD',n,s)
  if (kind in ['BinaryOperator','CompoundAssignOperator'] and n.get('opcode') in ['=','+=','-=','*=','/=','%=','&=','|=','^=','<<=','>>=']) or (kind=='UnaryOperator' and n.get('opcode') in ['++','--']):
   self.emit('WRITE',n,s);s=self.invalidate(s,'WRITE_MAY_INVALIDATE_CURRENT_GUARD_VALUES')
  return s
 def statement(self,n,states):
  kind=n.get('kind');cs=children(n)
  if kind=='CompoundStmt':
   s=clone(states)
   for c in cs:s=self.statement(c,s)
   return s
  if kind=='IfStmt':
   count=3 if n.get('hasElse') else 2
   if len(cs)!=count:return self.unsupported(n,states,'IF_LAYOUT_UNKNOWN')
   condition=self.expr(cs[0]);s=self.flow_expr(cs[0],states)
   if any(x.get('kind')=='CallExpr' or (x.get('kind') in ['BinaryOperator','CompoundAssignOperator','UnaryOperator'] and x.get('opcode') in ['=','+=','-=','++','--']) for x in old.v1.walk(cs[0])):
    self.note('SIDE_EFFECTING_CONDITION_VALUE_RELATION_UNKNOWN')
    for q in s:q['unknown']=True
   self.emit('BRANCH',cs[0],s,condition)
   a=self.statement(cs[1],self.branch(s,condition,True,n.get('range',{})))
   b=self.statement(cs[2],self.branch(s,condition,False,n.get('range',{}))) if count==3 else self.branch(s,condition,False,n.get('range',{}))
   return self.states_unique(a+b)
  if kind=='ReturnStmt':
   s=clone(states)
   for c in cs:s=self.flow_expr(c,s)
   self.emit('RETURN',n,s,self.expr(cs[0]) if cs else ['EMPTY',0,0,[]]);return []
  if kind in ['ForStmt','WhileStmt','DoStmt','SwitchStmt','GotoStmt','IndirectGotoStmt','LabelStmt','CaseStmt','DefaultStmt','BreakStmt','ContinueStmt','GCCAsmStmt']:
   return self.unsupported(n,states,'UNSUPPORTED_CONTROL_'+kind)
  if kind=='DeclStmt':
   s=clone(states)
   for c in cs:
    if c.get('kind')!='VarDecl':self.note('DECLARATION_KIND_UNKNOWN');continue
    values=children(c)
    if values:
     for v in values:s=self.flow_expr(v,s)
     target=['LOCAL',old.v1.type_tag(clean(c)),self.symbols[c['id']][1],[]]
     self.emit('INIT',c,s,['ASSIGN',target[1],0,[target,self.expr(values[-1])]])
     s=self.invalidate(s,'INITIALIZATION_MAY_INVALIDATE_CURRENT_GUARD_VALUES')
   return s
  if kind in ['NullStmt',None]:return clone(states)
  return self.flow_expr(n,states)
 def unsupported(self,n,states,reason):
  self.note(reason);s=clone(states)
  if n.get('kind') in ['LabelStmt','CaseStmt','DefaultStmt']:
   s=[dict(guards=[],unknown=True)] # possible non-lexical entry; never infer dead from preceding return
  for q in s:q['unknown']=True
  self.emit('UNKNOWN_CONTROL',n,s,['UNKNOWN',0,0,[]])
  # Descendant exits do not remove outer continuation: loops/gotos are not solved.
  for c in children(n):self.statement(c,clone(s))
  return self.invalidate(s,'UNKNOWN_CONTROL_MAY_INVALIDATE_GUARD_VALUES')
 def run(self):
  for name in self.names:
   if name not in self.funcs:continue
   self.name=name;self.owner=self.indices[name];self.ordinal=0;self.un=set();self.global_slots={}
   fn=self.funcs[name];params=[x for x in children(fn) if x['kind']=='ParmVarDecl'];body=next(x for x in children(fn) if x['kind']=='CompoundStmt')
   local=[x for x in old.v1.walk(body) if x.get('kind')=='VarDecl']
   self.symbols={x['id']:('PARAM',i) for i,x in enumerate(params)};self.symbols.update({x['id']:('LOCAL',i) for i,x in enumerate(local)})
   self.statement(body,[dict(guards=[],unknown=False)])
   self.function_unknown[str(self.owner)]=sorted(self.un)
  return self.pack()
 def pack(self):
  def zeros(n,k=None):return [0]*n if k is None else [[0]*k for _ in range(n)]
  t=dict(event_mask=zeros(LIMITS['events']),event_features=zeros(LIMITS['events'],6),event_path_mask=zeros(LIMITS['events'],LIMITS['alternatives']),event_paths=zeros(LIMITS['events'],LIMITS['alternatives']),expression_mask=zeros(LIMITS['expressions']),expression_token_mask=zeros(LIMITS['expressions'],LIMITS['expression_tokens']),expression_tokens=[zeros(LIMITS['expression_tokens'],4) for _ in range(LIMITS['expressions'])],predicate_mask=zeros(LIMITS['predicates']),predicate_features=zeros(LIMITS['predicates'],2),path_mask=zeros(LIMITS['paths']),path_unknown=zeros(LIMITS['paths']),path_guard_mask=zeros(LIMITS['paths'],LIMITS['guards']),path_guards=[zeros(LIMITS['guards'],2) for _ in range(LIMITS['paths'])],format_mask=zeros(LIMITS['formats']),format_features=zeros(LIMITS['formats'],11))
  for i,e in enumerate(self.events):
   t['event_mask'][i]=1;t['event_features'][i]=[KINDS[e['kind']],e['owner'],e['ordinal'],e['expression'],e['callee'],int(e['syntactically_unreachable'])]
   for j,p in enumerate(e['paths']):t['event_path_mask'][i][j]=1;t['event_paths'][i][j]=p
  for i,e in enumerate(self.expressions):
   ts=expression_tokens(e);t['expression_mask'][i]=1
   for j,v in enumerate(ts):t['expression_token_mask'][i][j]=1;t['expression_tokens'][i][j]=v
  for i,p in enumerate(self.predicates):t['predicate_mask'][i]=1;t['predicate_features'][i]=[p['owner'],p['expression']]
  for i,p in enumerate(self.paths):
   t['path_mask'][i]=1;t['path_unknown'][i]=int(p['unknown'])
   for j,g in enumerate(p['guards']):t['path_guard_mask'][i][j]=1;t['path_guards'][i][j]=g
  for i,f in enumerate(self.formats):
   t['format_mask'][i]=1;t['format_features'][i]=[f['call_event'],f['family'],CONV.get(f['conv'],0),LENGTH.get(f['length'],0),{'value':1,'width':2,'precision':3,'escaped':4,'unknown_format':5,'no_conversion':6}[f['role']],f['argument_index'],FT[f['expected']],FT[f['actual']],int(f['present']),int(f['known']),{'UNKNOWN':0,'TYPE_EQUAL':1,'TYPE_DIFF':2,'MISSING_ARGUMENT':3}[f['relation']]]
  return t
 def facts(self):return dict(events=self.events,expressions=self.expressions,predicates=self.predicates,paths=self.paths,formats=self.formats,unknown_reasons=sorted(self.unknown),function_unknown=self.function_unknown,flow_scope='STRUCTURED_STRAIGHT_LINE_IF_ELSE_RETURN_SHORT_CIRCUIT; PREDICATES_AT_EVALUATION_TIME',call_effects='CONSERVATIVE_GUARD_VALUE_INVALIDATION; NO_INTERPROCEDURAL_EFFECT_PROOF',path_feasibility='NOT_SOLVED',safety_verdict='NOT_COMPUTED',labels_used=False)

def extract(ast,focal,definitions):
 legacy=old.extract(ast,focal,definitions)
 if legacy.get('tensors') is None:return dict(schema=SCHEMA,status='QUARANTINED_LEGACY_CAP',reasons=legacy.get('reasons',[]),legacy=legacy,tensors=None,facts={},encoder_connected=False)
 x=Extractor(ast,focal,definitions,legacy)
 try:t=x.run();status='MATERIALIZED_PARTIAL' if x.unknown else 'MATERIALIZED';reason=[]
 except Cap as e:t=None;status='QUARANTINED_CAP';reason=[str(e)]
 return dict(schema=SCHEMA,status=status,reasons=reason,legacy=legacy,tensors=t,tensor_sha256=old.v1.digest(t) if t is not None else None,facts=x.facts(),encoder_connected=False)
def schema():return dict(schema=SCHEMA,limits=LIMITS,opcodes=OP,event_kinds=KINDS,format_type_codes=FT,conversion_codes=CONV,length_codes=LENGTH,abi='x86_64_linux_LP64_Clang14',legacy='V2 unchanged in separate legacy.tensors',new_encoder_connected=False,unknown_is_not_safe=True,not_supported=['general CFG','loop bounds','goto targets','alias proof','cross-TU calls','indirect call resolution','nonliteral format propagation','POSIX positional formats','full printf value-dependent type validity','model inference'])
