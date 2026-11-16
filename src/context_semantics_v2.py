"""Label-free, bounded call/guard and va_list event representation, candidate v2.

This is a source-fact extractor, not a safety checker or neural predictor.
V1 fields remain intact. New fields are not consumed by existing checkpoints.
"""
import copy,struct
import context_facts as v1

SCHEMA='bsfx-call-semantics-v2-candidate'
MAX_EVENTS=128;MAX_GUARDS=4;MAX_EXPR=32;MAX_ALIGN=64
OP={n:i for i,n in enumerate(['PAD','UNKNOWN','PARAM','LOCAL','INT','FLOAT','CAST','NOT','NEG','POS','ADDR','DEREF','BITNOT','ADD','SUB','MUL','DIV','MOD','LT','LE','EQ','NE','AND','OR','BITAND','BITOR','XOR','SHL','SHR','ASSIGN','CALL','COND','COMMA','INDEX'])}
BIN={'+':'ADD','-':'SUB','*':'MUL','/':'DIV','%':'MOD','<':'LT','<=':'LE','==':'EQ','!=':'NE','&&':'AND','||':'OR','&':'BITAND','|':'BITOR','^':'XOR','<<':'SHL','>>':'SHR','=':'ASSIGN',',':'COMMA'}
UN={'!':'NOT','-':'NEG','+':'POS','&':'ADDR','*':'DEREF','~':'BITNOT'}
KIND={'CALL':1,'START':2,'ARG':3,'END':4,'COPY':5}
BUILTIN={'__builtin_va_start':'START','__builtin_va_end':'END','__builtin_va_copy':'COPY'}
TRIVIAL_CAST={'LValueToRValue','ArrayToPointerDecay','FunctionToPointerDecay','BuiltinFnToFnPtr','NoOp'}

def strip(n):
    while n.get('kind') in ['ParenExpr','ConstantExpr'] or (n.get('kind')=='ImplicitCastExpr' and n.get('castKind') in TRIVIAL_CAST):
        if len(n.get('inner',[]))!=1:break
        n=n['inner'][0]
    return n

def direct(n):
    n=strip(n);r=n.get('referencedDecl',{})
    return r if n.get('kind')=='DeclRefExpr' and r.get('kind')=='FunctionDecl' else None

def expression(n,symbols,node_index,unknown):
    n=strip(n);kind=n.get('kind');children=n.get('inner',[]);typ=v1.type_tag(v1.clean_type(n))
    def e(c):return expression(c,symbols,node_index,unknown)
    def make(op,payload=0,args=(),type_id=typ):return [op,type_id,payload,list(args)]
    if kind=='DeclRefExpr':
        ref=n.get('referencedDecl',{});symbol=symbols.get(ref.get('id'))
        if symbol:return make('PARAM' if symbol[0]==1 else 'LOCAL',symbol[1])
        unknown.add('EXTERNAL_OR_UNRESOLVED_SYMBOL');return make('UNKNOWN')
    if kind in ['IntegerLiteral','CharacterLiteral']:
        try:
            value=int(n['value']);assert -(2**63)<=value<2**63
            return make('INT',value)
        except (ValueError,KeyError,AssertionError):unknown.add('INTEGER_RANGE');return make('UNKNOWN')
    if kind=='FloatingLiteral':
        try:return make('FLOAT',struct.unpack('<q',struct.pack('<d',float(n['value'])))[0])
        except (ValueError,KeyError,OverflowError):unknown.add('FLOAT_LITERAL');return make('UNKNOWN')
    if kind in ['CStyleCastExpr','ImplicitCastExpr'] and len(children)==1:return make('CAST',args=[e(children[0])])
    if kind=='UnaryOperator' and len(children)==1 and n.get('opcode') in UN:
        a=e(children[0]);op=UN[n['opcode']]
        if op in ['NEG','POS'] and a[0]=='INT':return make('INT',-a[2] if op=='NEG' else a[2])
        return make(op,args=[a])
    if kind in ['BinaryOperator','CompoundAssignOperator'] and len(children)==2:
        op=n.get('opcode');a,b=e(children[0]),e(children[1])
        if op in ['>','>=']:a,b=b,a;op='<' if op=='>' else '<='
        if op=='+' and b[0]=='INT' and b[2]<0:b=copy.deepcopy(b);b[2]=-b[2];op='-'
        if op in BIN:return make(BIN[op],args=[a,b])
    if kind=='ArraySubscriptExpr' and len(children)==2:return make('INDEX',args=list(map(e,children)))
    if kind=='ConditionalOperator' and len(children)==3:return make('COND',args=list(map(e,children)))
    if kind=='CallExpr' and children:
        ref=direct(children[0]);return make('CALL',node_index.get(ref.get('name'),-1) if ref else -1,[e(c) for c in children[1:]])
    unknown.add('EXPRESSION_'+str(kind));return make('UNKNOWN')

def negate(e):
    e=copy.deepcopy(e);op,typ,val,args=e
    if op=='INT':return ['INT',typ,int(not val),[]]
    if op=='NOT':return args[0]
    if op in ['LT','LE']:return ['LE' if op=='LT' else 'LT',typ,0,[args[1],args[0]]]
    if op in ['EQ','NE']:return ['NE' if op=='EQ' else 'EQ',typ,0,args]
    return ['NOT',typ,0,[e]]

def flat(e):
    op,typ,value,args=e;return [[OP[op],typ,value,len(args)]]+[r for a in args for r in flat(a)]

def extract(ast,focal,definitions):
    base=v1.extract(ast,focal,definitions)
    if base['tensors'] is None:return dict(schema=SCHEMA,status=base['status'],reasons=base['reasons'],tensors=None,base=base)
    names=[n['name'] for n in base['facts']['nodes']];indices={n:i for i,n in enumerate(names)}
    funcs={f['name']:f for f in definitions};symbols={};events=[];unknown=set();warnings=[];function_unknown={};by_owner={}
    for name in names:
        if name not in funcs:continue
        fn=funcs[name];params=[n for n in fn.get('inner',[]) if n.get('kind')=='ParmVarDecl'];body=next(n for n in fn['inner'] if n.get('kind')=='CompoundStmt')
        local=[n for n in v1.walk(body) if n.get('kind')=='VarDecl'];sy={n['id']:(1,i) for i,n in enumerate(params)};sy.update({n['id']:(2,i) for i,n in enumerate(local)});symbols[name]=sy
        un=set();owner=indices[name];owner_events=[]
        def expr(n):return expression(n,sy,indices,un)
        def obj(n):
            n=strip(n);ref=n.get('referencedDecl',{})
            if n.get('kind')=='DeclRefExpr' and ref.get('id') in sy:return list(sy[ref['id']])
            return [0,-1]
        def visit(n,guards=(),uncertain=False):
            kind=n.get('kind');cs=n.get('inner',[])
            if kind=='IfStmt':
                count=3 if n.get('hasElse') else 2
                if len(cs)!=count:un.add('IF_LAYOUT');uncertain=True
                else:
                    cond=expr(cs[0]);visit(cs[0],guards,uncertain);visit(cs[1],guards+(cond,),uncertain)
                    if count==3:visit(cs[2],guards+(negate(cond),),uncertain)
                    return
            if kind in ['WhileStmt','DoStmt','ForStmt','SwitchStmt','GotoStmt','IndirectGotoStmt','ConditionalOperator'] or (kind=='BinaryOperator' and n.get('opcode') in ['&&','||']):
                un.add('CONTROL_'+kind);uncertain=True
            if kind=='ReturnStmt':
                # Do not assert later lexical events execute: early-return CFG is not implemented.
                un.add('RETURN_FLOW_NOT_PROVEN')
            if kind in ['CallExpr','VAArgExpr']:
                ref=direct(cs[0]) if kind=='CallExpr' and cs else None;target=ref.get('name') if ref else None
                tag='ARG' if kind=='VAArgExpr' else BUILTIN.get(target,'CALL');args=cs[1:] if kind=='CallExpr' else cs[:1]
                nested=any(x.get('kind') in ['CallExpr','VAArgExpr'] for a in args for x in v1.walk(a))
                if nested:un.add('NESTED_EVENT_ORDER');uncertain=True
                objs=[obj(a) for a in args];eid=len(events)
                ev=dict(id=eid,owner=owner,owner_name=name,kind=tag,callee=indices.get(target,-1),callee_name=target,lexical_ordinal=len(owner_events),guards=list(guards),control_unknown=bool(uncertain),argument_expressions=[expr(a) for a in args],argument_types=[v1.type_tag(v1.clean_type(a)) for a in args],argument_objects=objs,object=objs[0] if tag in ['START','ARG','END','COPY'] and objs else [0,-1],copy_source=objs[1] if tag=='COPY' and len(objs)>1 else [0,-1],consumed_type=v1.type_tag(v1.clean_type(n)) if tag=='ARG' else 0)
                if tag in ['START','ARG','END','COPY'] and ev['object'][0]==0:un.add('OBJECT_ALIAS_UNRESOLVED')
                if tag=='CALL' and any(o[0]==0 and t in [v1.TYPE_VOCAB['POINTER'],v1.TYPE_VOCAB['ARRAY']] for o,t in zip(objs,ev['argument_types'])):un.add('POINTER_ARGUMENT_ALIAS_UNRESOLVED')
                events.append(ev);owner_events.append(eid)
            for c in cs:visit(c,guards,uncertain)
        visit(body);by_owner[owner]=owner_events;function_unknown[owner]=sorted(un);unknown.update(un)
    if len(events)>MAX_EVENTS:warnings.append('EVENT_CAP')
    for e in events:
        if len(e['guards'])>MAX_GUARDS:warnings.append('GUARD_CAP')
        if len(e['argument_expressions'])>v1.MAX_ARGS:warnings.append('EXPRESSION_ARGUMENT_CAP')
        if any(len(flat(x))>MAX_EXPR for x in e['guards']+e['argument_expressions']):warnings.append('EXPRESSION_TOKEN_CAP')
    # Direct actual-variable -> formal-parameter binding, with no pointer-alias inference.
    bindings=[]
    for e in events:
        if e['kind']!='CALL' or e['callee_name'] not in funcs:continue
        pars=[x for x in funcs[e['callee_name']]['inner'] if x.get('kind')=='ParmVarDecl']
        for i,objref in enumerate(e['argument_objects'][:len(pars)]):
            bindings.append(dict(call_event=e['id'],caller=e['owner'],actual_object=objref,callee=e['callee'],formal_kind=1,formal_slot=i,known=objref[0]!=0))
    if len(bindings)>MAX_ALIGN:warnings.append('BINDING_CAP')
    # Optional straight-line syntactic consumption expansion. It does not decide C safety.
    # After a callee consumes a shared object, subsequent caller consumption has unknown slot.
    align=[];trace_notes=[]
    for root in events:
        if root['kind']!='CALL' or root['callee_name'] not in funcs or not funcs[root['callee_name']].get('variadic'):continue
        target=root['callee'];fixed=len([p for p in funcs[root['callee_name']]['inner'] if p.get('kind')=='ParmVarDecl']);provided=root['argument_types'][fixed:]
        if root['guards'] or root['control_unknown']:trace_notes.append(dict(root_event=root['id'],reason='GUARDED_ROOT'));continue
        def trace(owner,env,stack):
            if owner in stack or len(stack)>=4:trace_notes.append(dict(root_event=root['id'],reason='RECURSION_OR_DEPTH'));return False
            if function_unknown.get(owner):trace_notes.append(dict(root_event=root['id'],owner=owner,reason='FUNCTION_FLOW_PARTIAL'));return False
            ids=by_owner.get(owner)
            if ids is None:trace_notes.append(dict(root_event=root['id'],owner=owner,reason='EXTERNAL_BODY'));return False
            if any(events[i]['guards'] or events[i]['control_unknown'] for i in ids):trace_notes.append(dict(root_event=root['id'],owner=owner,reason='GUARDED_EVENTS'));return False
            for eid in ids:
                e=events[eid];key=tuple(e['object']);tag=e['kind']
                if tag=='START':
                    if owner==target and key[0]:env[key]=dict(cursor=0,known=True,consumed=0)
                elif tag=='COPY':
                    src=env.get(tuple(e['copy_source']))
                    if src and key[0]:env[key]=dict(src)
                elif tag=='ARG':
                    state=env.get(key);known=bool(state and state['known']);slot=state['cursor'] if known else -1
                    align.append(dict(root_event=root['id'],consumer_event=eid,object_owner=owner,object_kind=key[0],object_slot=key[1],variadic_slot=slot,consumed_type=e['consumed_type'],actual_type=provided[slot] if known and slot<len(provided) else 0,slot_known=known,argument_present=bool(known and slot<len(provided))))
                    if state:state['cursor']+=1;state['consumed']+=1
                elif tag=='END':
                    if key in env:env[key]['known']=False
                elif tag=='CALL':
                    if e['callee_name'] not in funcs:
                        exposed=[env.get(tuple(o)) for o in e['argument_objects']]
                        if any(s is not None for s in exposed):
                            for state in exposed:
                                if state is not None:state['known']=False
                            trace_notes.append(dict(root_event=root['id'],call_event=eid,reason='EXTERNAL_OR_INDIRECT_OBJECT_CONSUMPTION_UNKNOWN'))
                        continue
                    passed={};parent_states=[]
                    for b in bindings:
                        if b['call_event']==eid:
                            state=env.get(tuple(b['actual_object']))
                            if state is not None:passed[(1,b['formal_slot'])]=dict(state);parent_states.append((state,passed[(1,b['formal_slot'])]))
                    if passed:
                        ok=trace(e['callee'],passed,stack+[owner])
                        for original,child in parent_states:
                            if not ok or child['consumed']>original['consumed']:original['known']=False
            return True
        trace(target,{},[])
    if len(align)>MAX_ALIGN:warnings.append('CONSUMPTION_ALIGNMENT_CAP')
    facts=dict(events=events,actual_to_formal_bindings=bindings,consumption_alignment=align,trace_notes=trace_notes,function_partial_reasons=function_unknown,unknown_reasons=sorted(unknown),event_order='LEXICAL_AST_ORDER_NOT_GENERAL_EXECUTION_TRACE',call_resolution='DIRECT_SAME_TRANSLATION_UNIT_ONLY',alignment_scope='UNGUARDED_STRAIGHT_LINE_DIRECT_BODY; POST_CALLEE_CONSUMPTION_SLOT_UNKNOWN',labels_used=False)
    if warnings:return dict(schema=SCHEMA,status='QUARANTINED_CAP',reasons=sorted(set(warnings)),facts=facts,base=base,tensors=None)
    # Numeric values depend on syntax/types/ordinal links, never names, sample IDs or labels.
    t={'v1__'+k:copy.deepcopy(v) for k,v in base['tensors'].items()}
    t.update(event_mask=[0]*MAX_EVENTS,event_features=[[0]*12 for _ in range(MAX_EVENTS)],guard_mask=[[0]*MAX_GUARDS for _ in range(MAX_EVENTS)],guard_tokens=[[[[0]*4 for _ in range(MAX_EXPR)] for _ in range(MAX_GUARDS)] for _ in range(MAX_EVENTS)],argument_mask=[[0]*v1.MAX_ARGS for _ in range(MAX_EVENTS)],argument_tokens=[[[[0]*4 for _ in range(MAX_EXPR)] for _ in range(v1.MAX_ARGS)] for _ in range(MAX_EVENTS)],binding_mask=[0]*MAX_ALIGN,binding_features=[[0]*8 for _ in range(MAX_ALIGN)],alignment_mask=[0]*MAX_ALIGN,alignment_features=[[0]*10 for _ in range(MAX_ALIGN)],function_partial=[0]*v1.MAX_NODES)
    previous={}
    for e in events:
        i=e['id'];ob=e['object'];key=(e['owner'],*ob);prev=previous.get(key,-1) if ob[0] else -1
        # row12 = kind,owner,ordinal,target,objkind,objslot,copykind,copyslot,consumedtype,controlunknown,prevsameobject,constantfalseguard
        dead=any(g[0]=='INT' and g[2]==0 for g in e['guards'])
        t['event_mask'][i]=1;t['event_features'][i]=[KIND[e['kind']],e['owner'],e['lexical_ordinal'],e['callee'],*ob,*e['copy_source'],e['consumed_type'],int(e['control_unknown']),prev,int(dead)]
        if ob[0]:previous[key]=i
        for j,g in enumerate(e['guards']):
            t['guard_mask'][i][j]=1;f=flat(g);t['guard_tokens'][i][j][:len(f)]=f
        for j,a in enumerate(e['argument_expressions']):
            t['argument_mask'][i][j]=1;f=flat(a);t['argument_tokens'][i][j][:len(f)]=f
    for i,b in enumerate(bindings):t['binding_mask'][i]=1;t['binding_features'][i]=[b['call_event'],b['caller'],*b['actual_object'],b['callee'],b['formal_kind'],b['formal_slot'],int(b['known'])]
    for i,a in enumerate(align):t['alignment_mask'][i]=1;t['alignment_features'][i]=[a['root_event'],a['consumer_event'],a['object_owner'],a['object_kind'],a['object_slot'],a['variadic_slot'],a['consumed_type'],a['actual_type'],int(a['slot_known']),int(a['argument_present'])]
    for i,reasons in function_unknown.items():t['function_partial'][i]=int(bool(reasons))
    return dict(schema=SCHEMA,status='MATERIALIZED_PARTIAL' if unknown or trace_notes else 'MATERIALIZED',facts=facts,base=base,tensors=t,tensor_sha256=v1.digest(t),v1_tensor_sha256=base['tensor_sha256'],encoder_connected=False)

def schema():
    return dict(schema=SCHEMA,opcodes=OP,event_kinds=KIND,max_events=MAX_EVENTS,max_guards=MAX_GUARDS,max_expression_tokens=MAX_EXPR,max_bindings=MAX_ALIGN,max_alignments=MAX_ALIGN,expression_token_columns=['opcode','type_id','payload','arity'],object_kinds={'unknown':0,'formal_parameter':1,'local_declaration':2},event_columns=['kind','owner_node','lexical_ordinal','target_node','object_kind','object_slot','copy_source_kind','copy_source_slot','consumed_type','control_unknown','previous_lexical_same_object_event','constant_false_guard'],binding_columns=['call_event','caller','actual_kind','actual_slot','callee','formal_kind','formal_slot','known'],alignment_columns=['root_call_event','consumer_event','object_owner','object_kind','object_slot','variadic_slot','consumed_type','promoted_actual_type','slot_known','argument_present'],type_vocabulary=v1.TYPE_VOCAB,integer_dtype='int64',labels_or_identifiers_in_tensor=False,unknown_is_not_safe=True,encoder_connected=False,limitations=['no_general_CFG_or_alias_analysis','no_cross_translation_unit_resolution','no_arbitrary_expression_equivalence','no_path_feasibility_or_termination_proof','no_neural_performance_claim'])
