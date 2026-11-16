"""Verify public aggregates only. Does not load models or private records."""
from pathlib import Path
import ast,hashlib,json,math,statistics
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10),(a,b)
def main():
    code=read(ROOT/'provenance/CODE_VERSION.json')
    for row in code['code_files']:
        p=ROOT/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        ast.parse(p.read_text(encoding='utf-8'))
    assert len(code['bound_modules'])==48
    combined={};checked=0
    for path in [ROOT/'results/component_ablation/ALL_METRICS.json',ROOT/'results/relation_ablation/ALL_METRICS.json']:
        for name,row in read(path).items():
            if name in combined:assert row==combined[name]
            combined[name]=row
            for split in ['train','validation','test']:
                for mode in ['fixed','calibrated']:
                    m=row[split+'_'+mode];tp,tn,fp,fn=[m[k] for k in ['TP','TN','FP','FN']]
                    assert m['n']==tp+tn+fp+fn and m['positive']==tp+fn and m['negative']==tn+fp
                    div=lambda a,b:a/b if b else 0.
                    f1=div(2*tp,2*tp+fp+fn);nf1=div(2*tn,2*tn+fp+fn)
                    for key,value in {'accuracy':(tp+tn)/m['n'],'precision':div(tp,tp+fp),'recall':div(tp,tp+fn),
                                      'F1':f1,'negative_F1':nf1,'macro_F1':(f1+nf1)/2}.items():close(m[key],value)
                    assert 0<=m['AUROC']<=1 and 0<=m['AP']<=1 and m['mean_CE']>=0
                    checked+=1
    assert len(combined)==48
    summary=read(ROOT/'results/AGGREGATE_SUMMARY.json')
    for arm,rec in summary.items():
        rows=[v for v in combined.values() if v['arm']['ablation']==arm]
        assert len(rows)==rec['n_runs']==3
        for mode in ['train_calibrated','validation_calibrated','test_fixed','test_calibrated']:
            for metric,val in rec[mode].items():
                close(val['mean'],statistics.mean(x[mode][metric] for x in rows))
                close(val['sample_sd'],statistics.stdev(x[mode][metric] for x in rows))
    result={'status':'PASS','code_files_sha_verified':len(code['code_files']),
       'distinct_current_config_seed_results':len(combined),'classification_metric_dicts_checked':checked,
       'scope':'Aggregate arithmetic, SHA and syntax only. AUC/AP/CE cannot be independently recomputed without private per-sample records.',
       'model_forwards':0,'checkpoint_loads':0,'optimizer_updates':0}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
