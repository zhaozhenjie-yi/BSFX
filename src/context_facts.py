"""Bounded C call-context facts from Clang JSON AST. No label or learner access."""
from __future__ import annotations
import collections,hashlib,json,re

SCHEMA='bsfx-c-call-context-v1-candidate'
MAX_NODES=12;MAX_CALLS=32;MAX_ARGS=8;MAX_VA_ARGS=8
TYPE_VOCAB={s:i for i,s in enumerate(['PAD','UNKNOWN','void','_Bool','char','signed char','unsigned char',
    'short','unsigned short','int','unsigned int','long','unsigned long','long long','unsigned long long',
    'float','double','long double','POINTER','ARRAY','RECORD','FUNCTION'])}

def walk(node):
    if not isinstance(node,dict):return
    yield node
    for child in node.get('inner',[]):yield from walk(child)

def clean_type(node):
    value=node.get('type',{});text=value.get('desugaredQualType',value.get('qualType',''))
    return re.sub(r'\s+',' ',re.sub(r'\b(const|volatile|restrict|__restrict)\b','',text)).strip()

def type_tag(text):
    aliases={'signed':'int','signed int':'int','unsigned':'unsigned int','short int':'short',
             'signed short':'short','signed short int':'short','unsigned short int':'unsigned short',
             'long int':'long','signed long':'long','signed long int':'long','unsigned long int':'unsigned long',
             'long long int':'long long','unsigned long long int':'unsigned long long'}
    text=aliases.get(text,text)
    if '*' in text:return TYPE_VOCAB['POINTER']
    if '[' in text:return TYPE_VOCAB['ARRAY']
    if '(' in text:return TYPE_VOCAB['FUNCTION']
    if text.startswith(('struct ','union ')):return TYPE_VOCAB['RECORD']
    return TYPE_VOCAB.get(text,TYPE_VOCAB['UNKNOWN'])

def direct_reference(expr):
    while expr.get('kind') in ['ImplicitCastExpr','ParenExpr','ConstantExpr'] and len(expr.get('inner',[]))==1:
        expr=expr['inner'][0]
    ref=expr.get('referencedDecl',{})
    return ref if expr.get('kind')=='DeclRefExpr' and ref.get('kind')=='FunctionDecl' else None

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def extract(ast,focal_name,source_definitions):
    """source_definitions are already restricted to active main-source definitions."""
    definitions=collections.defaultdict(list)
    for fn in source_definitions:definitions[fn['name']].append(fn)
    if len(definitions.get(focal_name,[]))!=1:raise ValueError('focal definition must be unique')
    if any(len(v)>1 for v in definitions.values()):raise ValueError('ambiguous same-name source definitions')
    bodies={name:xs[0] for name,xs in definitions.items()}
    declarations={x['id']:x for x in walk(ast) if x.get('kind')=='FunctionDecl' and x.get('id')}
    refs={};calls=[];per_function={}
    for name,fn in bodies.items():
        body=next(x for x in fn.get('inner',[]) if x.get('kind')=='CompoundStmt')
        events=[];va=[]
        for node in walk(body):
            if node.get('kind')=='VAArgExpr':va.append(clean_type(node))
            if node.get('kind')!='CallExpr':continue
            children=node.get('inner',[])
            ref=direct_reference(children[0]) if children else None
            target=ref.get('name') if ref else None
            if ref:refs.setdefault(target,declarations.get(ref.get('id'),ref))
            args=[clean_type(arg) for arg in children[1:]]
            item=dict(caller=name,callee=target,resolution='AST_DIRECT_FUNCTION_REF' if ref else 'UNRESOLVED_INDIRECT',
                argument_types=args,argument_type_ids=[type_tag(a) for a in args],ordinal=len(events),
                location=node.get('range',{}),callee_ast_id=ref.get('id') if ref else None)
            events.append(item);calls.append(item)
        per_function[name]=dict(calls=events,va_arg_types=va)
    # Focal node first. Immediate outgoing callees follow call occurrence; then direct callers in source order.
    order=[focal_name]
    for call in per_function[focal_name]['calls']:
        if call['callee'] and call['callee'] not in order:order.append(call['callee'])
    for name in bodies:
        if any(x['callee']==focal_name for x in per_function[name]['calls']) and name not in order:order.append(name)
    selected=[c for c in calls if c['caller'] in order and (c['callee'] in order or c['callee'] is None)]
    selected.sort(key=lambda c:(order.index(c['caller']),c['ordinal']))
    scope_omitted=[c for c in calls if c['caller'] in order and c['callee'] is not None and c['callee'] not in order]
    warnings=[]
    if len(order)>MAX_NODES:warnings.append('NODE_CAP_EXCEEDED')
    if len(selected)>MAX_CALLS:warnings.append('CALL_CAP_EXCEEDED')
    if any(len(c['argument_types'])>MAX_ARGS for c in selected):warnings.append('ARGUMENT_CAP_EXCEEDED')
    nodefacts=[]
    for name in order:
        decl=bodies.get(name,refs.get(name,{}));params=[x for x in decl.get('inner',[]) if x.get('kind')=='ParmVarDecl']
        param_types=[clean_type(p) for p in params]
        ft=clean_type(decl);ret=ft.split('(',1)[0].strip()
        va=per_function.get(name,{}).get('va_arg_types',[])
        if len(params)>MAX_ARGS:warnings.append('FORMAL_CAP_EXCEEDED')
        if len(va)>MAX_VA_ARGS:warnings.append('VA_ARG_CAP_EXCEEDED')
        nodefacts.append(dict(name=name,source_definition_available=name in bodies,variadic=bool(decl.get('variadic',False) or '...' in ft),
            fixed_parameter_types=param_types,return_type=ret,va_arg_consumed_types=va,
            signature_available=bool(params) or '(void)' in ft or bool(decl.get('inner')),
            unresolved_call_count=sum(c['callee'] is None for c in per_function.get(name,{}).get('calls',[]))))
    facts=dict(schema=SCHEMA,focal_function=focal_name,nodes=nodefacts,calls=selected,
        omitted_neighbour_outgoing_edges=scope_omitted,scope='FOCAL_PLUS_IMMEDIATE_DIRECT_CALLERS_AND_CALLEES_WITHIN_ACTIVE_TRANSLATION_UNIT',
        unproven=['cross_translation_unit_resolution','indirect_call_targets','path_feasibility','termination','business_contract_safety'])
    if warnings:return dict(status='QUARANTINED_CAP',reasons=sorted(set(warnings)),facts=facts,tensors=None,tensor_sha256=None)
    index={name:i for i,name in enumerate(order)}
    tensors=dict(node_mask=[0]*MAX_NODES,node_features=[[0]*6 for _ in range(MAX_NODES)],
        fixed_parameter_types=[[0]*MAX_ARGS for _ in range(MAX_NODES)],fixed_parameter_mask=[[0]*MAX_ARGS for _ in range(MAX_NODES)],
        va_arg_types=[[0]*MAX_VA_ARGS for _ in range(MAX_NODES)],va_arg_mask=[[0]*MAX_VA_ARGS for _ in range(MAX_NODES)],
        call_mask=[0]*MAX_CALLS,call_source=[-1]*MAX_CALLS,call_target=[-1]*MAX_CALLS,
        call_resolution=[0]*MAX_CALLS,call_argument_types=[[0]*MAX_ARGS for _ in range(MAX_CALLS)],
        call_argument_mask=[[0]*MAX_ARGS for _ in range(MAX_CALLS)],
        # Omitted-neighbour edge counts express incomplete scope, not assumed safety.
        omitted_neighbour_edge_count=[0]*MAX_NODES)
    for i,n in enumerate(nodefacts):
        tensors['node_mask'][i]=1
        tensors['node_features'][i]=[int(i==0),int(n['source_definition_available']),int(n['variadic']),len(n['fixed_parameter_types']),type_tag(n['return_type']),int(n['signature_available'])]
        for j,p in enumerate(n['fixed_parameter_types']):tensors['fixed_parameter_types'][i][j]=type_tag(p);tensors['fixed_parameter_mask'][i][j]=1
        for j,p in enumerate(n['va_arg_consumed_types']):tensors['va_arg_types'][i][j]=type_tag(p);tensors['va_arg_mask'][i][j]=1
        tensors['omitted_neighbour_edge_count'][i]=sum(c['caller']==n['name'] for c in scope_omitted)
    for i,c in enumerate(selected):
        tensors['call_mask'][i]=1;tensors['call_source'][i]=index[c['caller']];tensors['call_target'][i]=index[c['callee']] if c['callee'] else -1
        tensors['call_resolution'][i]=1 if c['callee'] else 2
        for j,t in enumerate(c['argument_type_ids']):tensors['call_argument_types'][i][j]=t;tensors['call_argument_mask'][i][j]=1
    return dict(status='MATERIALIZED',facts=facts,tensors=tensors,tensor_sha256=digest(tensors),
        integer_dtype='int64',unknown_type_count=sum(t==TYPE_VOCAB['UNKNOWN'] for n in nodefacts for t in map(type_tag,n['fixed_parameter_types']+n['va_arg_consumed_types'])))

def schema():
    return dict(schema=SCHEMA,type_vocabulary=TYPE_VOCAB,max_nodes=MAX_NODES,max_calls=MAX_CALLS,max_arguments=MAX_ARGS,max_va_args=MAX_VA_ARGS,
        node_feature_columns=['is_focal','definition_available_in_source','variadic','fixed_parameter_count','return_type_tag','signature_available'],
        call_resolution_codes={'padding':0,'direct_function_reference':1,'unresolved_indirect':2},
        external_signature='May be known from declaration while body is absent',
        unsupported_type_policy='UNKNOWN; pointers/arrays/records coarse categories, no complete ABI/type equivalence claim',
        learner_integration='NOT_CONNECTED_TO_EXISTING_CHECKPOINTS',labels_or_names_in_tensors=False)
