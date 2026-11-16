import os,sys,json,pathlib,hashlib
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;ROOT=R.parent
F=ROOT/'automotive_preprocess_repair_20261023_r1';P=ROOT/'school_data_validation_20261017_150757'
A=ROOT/'context_learning_ablation_20261024_r1';Q=ROOT/'context_v2_learning_ablation_20261024_r1'
N=ROOT/'context_v3_encoder_20261025_r1';V=ROOT/'context_semantics_v3_20261025_r1'
sys.path[:0]=[str(R),str(F),str(Q),str(A),str(N),str(V),str(ROOT/'context_semantics_v2_20261024_r1'),
 str(ROOT/'context_visibility_regression_20261024_r1'),str(P),str(P/'runtime_source')]
import numpy as np,torch
from semantic_encoder import prepare_semantics,ContextV2Model
from semantic_encoder_v3 import prepare_v3,ContextV3Model
from evaluation_utils import tree_digest,digest,move
def read(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]
def save(p,x):
    with pathlib.Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load_np(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def prepared(sid,info):
    availability=info['availability']
    c2=prepare_semantics(load_np(R/'preparation04/tensors_v2'/(sid+'.npz'))) if availability['v2'] else dict(legacy=None,semantics=None)
    if availability['legacy'] and c2['legacy'] is None:
        c2['legacy']={k:torch.from_numpy(v.copy()).long() for k,v in load_np(R/'preparation04/tensors_legacy'/(sid+'.npz')).items()}
    c2['availability']=availability
    c3=prepare_v3(load_np(R/'preparation04/tensors_v3'/(sid+'.npz')),info['v3_status']) if availability['v3'] else None
    return dict(v2=c2,v3=c3)
