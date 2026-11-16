"""Conservative integer AST values for an explicitly LP64 analysis ABI.

No source-text eval, floating folding, hardware ABI inference, or side effects.
"""
import collections,re

def walk(n):
 if isinstance(n,dict):
  yield n
  for c in n.get('inner',[]):yield from walk(c)

class IntegerConstants:
 def __init__(self,ast,abi='x86_64_linux_LP64'):
  if abi!='x86_64_linux_LP64':raise ValueError('INTEGER_ABI_NOT_CONFIGURED')
  self.abi=abi;self.enums={};self.accepted=collections.Counter();self.refused=collections.Counter()
  for enum in walk(ast):
   if enum.get('kind')!='EnumDecl':continue
   previous=-1
   for child in enum.get('inner',[]):
    if child.get('kind')!='EnumConstantDecl':continue
    values=[x.get('value') for x in walk(child) if x.get('kind')=='ConstantExpr' and x.get('value') is not None]
    if not values:values=[x.get('value') for x in child.get('inner',[]) if x.get('kind')=='IntegerLiteral']
    if values:
     try:value=int(values[0])
     except (TypeError,ValueError):previous=None;continue
    elif not child.get('inner') and previous is not None:value=previous+1
    else:previous=None;continue
    if not -(2**63)<=value<2**63:previous=None;continue
    self.enums[child.get('id')]=value;previous=value
  # A cast back to a declared enum value is representable in every
  # compatible enum integer type. Do not guess the underlying type or width.
  self.enum_values={};self.enum_aliases={};self.enum_names={}
  for enum in walk(ast):
   if enum.get('kind')=='EnumDecl':
    values={self.enums[x['id']] for x in enum.get('inner',[]) if x.get('id') in self.enums}
    if values:
     self.enum_values[enum['id']]=values
     if enum.get('name'):self.enum_names['enum '+enum['name']]=enum['id']
  for decl in walk(ast):
   if decl.get('kind')=='TypedefDecl':
    targets=[x.get('ownedTagDecl',x.get('decl',{})).get('id') for x in walk(decl)]
    target=next((x for x in targets if x in self.enum_values),None)
    if target:self.enum_aliases[decl['id']]=target
 def spec(self,n):
  typ=n.get('type',{});t=typ.get('desugaredQualType',typ.get('qualType',''))
  t=re.sub(r'\b(const|volatile|restrict)\b','',t);t=' '.join(t.split())
  table={'_Bool':(1,False),'bool':(1,False),'char':(8,True),'signed char':(8,True),'unsigned char':(8,False),
   'short':(16,True),'short int':(16,True),'unsigned short':(16,False),'unsigned short int':(16,False),
   'int':(32,True),'signed int':(32,True),'unsigned int':(32,False),'unsigned':(32,False),
   'long':(64,True),'long int':(64,True),'unsigned long':(64,False),'unsigned long int':(64,False),
   'long long':(64,True),'long long int':(64,True),'unsigned long long':(64,False),'unsigned long long int':(64,False)}
  return table.get(t)
 def fit(self,value,spec,cast=False):
  if spec is None:return None
  bits,signed=spec
  if bits==1:return int(value!=0)
  if not signed:return value%(1<<bits)
  if -(1<<(bits-1))<=value<(1<<(bits-1)):return value
  self.refused['SIGNED_CAST_IMPLEMENTATION_DEFINED' if cast else 'SIGNED_OVERFLOW']+=1;return None
 def _eval(self,n):
  kind=n.get('kind');cs=n.get('inner',[]);spec=self.spec(n)
  if kind in ['IntegerLiteral','CharacterLiteral']:
   try:v=int(n['value'])
   except (KeyError,TypeError,ValueError):return None
   return self.fit(v,spec)
  if kind=='DeclRefExpr':
   ref=n.get('referencedDecl',{})
   if ref.get('kind')=='EnumConstantDecl' and ref.get('id') in self.enums:
    self.accepted['ENUM_VALUE']+=1;return self.enums[ref['id']]
   return None
  if kind in ['ParenExpr','ConstantExpr'] and len(cs)==1:return self._eval(cs[0])
  if kind in ['CStyleCastExpr','ImplicitCastExpr'] and len(cs)==1:
   if n.get('castKind') not in ['IntegralCast','IntegralToBoolean','NoOp','LValueToRValue']:return None
   v=self._eval(cs[0])
   if v is None:return None
   if spec is None:
    typ=n.get('type',{});target=self.enum_aliases.get(typ.get('typeAliasDeclId')) or self.enum_names.get(typ.get('qualType'))
    if target is not None and v in self.enum_values[target]:
     self.accepted['CAST_TO_DECLARED_ENUM_VALUE']+=1;return v
    return None
   return self.fit(v,spec,cast=True)
  if kind=='UnaryOperator' and len(cs)==1 and n.get('opcode') in ['+','-','~','!']:
   a=self._eval(cs[0])
   if a is None:return None
   op=n['opcode'];v=a if op=='+' else -a if op=='-' else ~a if op=='~' else int(not a)
   return self.fit(v,spec)
  if kind=='BinaryOperator' and len(cs)==2 and n.get('opcode') in ['+','-','*','/','%','<<','>>','&','|','^','==','!=','<','<=','>','>=','&&','||']:
   a,b=self._eval(cs[0]),self._eval(cs[1]);op=n['opcode']
   if a is None or b is None or spec is None:return None
   if op in ['/','%']:
    if b==0:self.refused['DIVISION_BY_ZERO']+=1;return None
    if spec[1] and a==-(1<<(spec[0]-1)) and b==-1:self.refused['SIGNED_OVERFLOW']+=1;return None
    q=(abs(a)//abs(b))*(-1 if (a<0)!=(b<0) else 1);v=q if op=='/' else a-q*b
   elif op in ['<<','>>']:
    if b<0 or b>=spec[0]:self.refused['SHIFT_RANGE']+=1;return None
    if a<0:self.refused['SIGNED_NEGATIVE_SHIFT']+=1;return None
    v=a<<b if op=='<<' else a>>b
   else:
    v={'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'&':lambda:a&b,'|':lambda:a|b,'^':lambda:a^b,
     '==':lambda:int(a==b),'!=':lambda:int(a!=b),'<':lambda:int(a<b),'<=':lambda:int(a<=b),'>':lambda:int(a>b),'>=':lambda:int(a>=b),
     '&&':lambda:int(bool(a) and bool(b)),'||':lambda:int(bool(a) or bool(b))}[op]()
   return self.fit(v,spec)
  return None
 def value(self,n):
  # A volatile read, call, assignment, pointer/float expression is never folded.
  if any(x.get('kind') in ['CallExpr','VAArgExpr','CompoundAssignOperator'] or 'volatile' in x.get('type',{}).get('qualType','') or
   (x.get('kind') in ['UnaryOperator','BinaryOperator'] and x.get('opcode') in ['++','--','=']) for x in walk(n)):return None
  v=self._eval(n)
  if v is not None:
   if not -(2**63)<=v<2**63:self.refused['INT64_PAYLOAD_RANGE']+=1;return None
   self.accepted['INTEGER_VALUE']+=1
  return v
 def audit(self):return dict(abi=self.abi,enum_definitions=len(self.enums),accepted=dict(self.accepted),refused=dict(self.refused),
  policy='Pure integer AST only; signed overflow/implementation-defined signed casts, invalid division/shifts stay unresolved. No target MCU ABI claim.')
