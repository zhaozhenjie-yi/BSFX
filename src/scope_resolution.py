"""Quarantine ambiguous immediate context; omit unrelated duplicate bodies.

No name rewriting or arbitrary choice between duplicate definitions. This only
admits a unique focal function whose immediate source callers/callees have
unique names. Outgoing calls beyond that scope remain outside the proof scope.
"""
from collections import defaultdict
def walk(n):
    if not isinstance(n,dict):return
    yield n
    for c in n.get('inner',[]):yield from walk(c)
def direct(n):
    while n.get('kind') in ['ImplicitCastExpr','ParenExpr','ConstantExpr'] and len(n.get('inner',[]))==1:n=n['inner'][0]
    ref=n.get('referencedDecl',{})
    return ref.get('name') if n.get('kind')=='DeclRefExpr' and ref.get('kind')=='FunctionDecl' else None
def calls(fn):
    return {direct(n['inner'][0]) for n in walk(fn) if n.get('kind')=='CallExpr' and n.get('inner')} - {None}
def select(focal,definitions):
    groups=defaultdict(list)
    for f in definitions:groups[f['name']].append(f)
    duplicates={name for name,items in groups.items() if len(items)>1}
    if len(groups.get(focal,[]))!=1:raise ValueError('FOCAL_SAME_NAME_AMBIGUITY_RETAINED')
    neighbours=(calls(groups[focal][0]) & set(groups)) | {f['name'] for f in definitions if focal in calls(f)}
    bad=duplicates & (neighbours | {focal})
    if bad:raise ValueError('IMMEDIATE_CONTEXT_SAME_NAME_AMBIGUITY_RETAINED:'+','.join(sorted(bad)))
    return [f for f in definitions if f['name'] not in duplicates],dict(excluded_unrelated_duplicate_names=sorted(duplicates),immediate_context_names=sorted(neighbours),focal_unique=True,scope='Unique focal and unique immediate source callers/callees only; no whole-TU identity proof')
