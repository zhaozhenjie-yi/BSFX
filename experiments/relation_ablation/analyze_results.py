"""Pure-Python, local recomputation and paired reporting from archived predictions."""
from pathlib import Path
import json,math,statistics,hashlib
R=Path(__file__).resolve().parent;S=R/'school_results';E=S/'final_evaluation'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def recompute(pred,t):
    tp=tn=fp=fn=0
    for x in pred:
        y=x['label'];z=x['probability']>t
        tp+=int(y==1 and z);tn+=int(y==0 and not z);fp+=int(y==0 and z);fn+=int(y==1 and not z)
    def div(a,b):return a/b if b else 0.
    f1=div(2*tp,2*tp+fp+fn);f0=div(2*tn,2*tn+fp+fn)
    pos=[x['probability'] for x in pred if x['label']];neg=[x['probability'] for x in pred if not x['label']]
    auc=sum(float(a>b)+.5*float(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
    ap=0.;previous=0
    for score in sorted({x['probability'] for x in pred},reverse=True):
        sel=[x for x in pred if x['probability']>=score];p=sum(x['label'] for x in sel)
        ap+=(p-previous)/len(pos)*p/len(sel);previous=p
    ce=[]
    for x in pred:
        a,b=x['logits'];m=max(a,b);ce.append(m+math.log(math.exp(a-m)+math.exp(b-m))-x['logits'][x['label']])
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(TP=tp,TN=tn,FP=fp,FN=fn,F1=f1,negative_F1=f0,macro_F1=(f1+f0)/2,accuracy=(tp+tn)/len(pred),
                precision=div(tp,tp+fp),recall=div(tp,tp+fn),negative_recall=div(tn,tn+fp),
                balanced_accuracy=(div(tp,tp+fn)+div(tn,tn+fp))/2,MCC=div(tp*tn-fp*fn,den),AUROC=auc,AP=ap,mean_CE=statistics.mean(ce))
def summary(v):return dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v),values=v)
def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-r,c+r]
cfg=read(S/'EXPERIMENT_CONFIG.json');allm=read(E/'ALL_METRICS.json');verified=0;predictions={}
for name,v in allm.items():
    predictions[name]={}
    for split in ['train','validation','test']:
        p=E/f'{name}_{split}.json' if v['reused'] or split=='test' else S/'runs'/name/f'{"training" if split=="train" else split}_predictions.json'
        pred=read(p);predictions[name][split]=pred
        for kind,t in [('fixed',.5),('calibrated',v['threshold'])]:
            own=recompute(pred,t);expected=v[f'{split}_{kind}']
            for k,x in own.items():assert abs(x-expected[k])<1e-10,(name,split,kind,k,x,expected[k])
            verified+=1
metrics=['F1','macro_F1','AUROC','AP','mean_CE','recall','negative_recall','balanced_accuracy','MCC','accuracy']
groups=['full']+list(cfg['ablations']);aggregate={};effects={};counts={}
for a in groups:
    keys=[f'{a}_seed{s}' for s in cfg['seeds']]
    aggregate[a]={f'{sp}_{ki}':{m:summary([allm[n][f'{sp}_{ki}'][m] for n in keys]) for m in metrics} for sp in ['train','validation','test'] for ki in ['fixed','calibrated']}
    if a!='full':
        effects[a]={f'{sp}_{ki}':{m:summary([allm[f'full_seed{s}'][f'{sp}_{ki}'][m]-allm[f'{a}_seed{s}'][f'{sp}_{ki}'][m] for s in cfg['seeds']]) for m in metrics} for sp in ['train','validation','test'] for ki in ['fixed','calibrated']}
    counts[a]={str(s):{k:allm[f'{a}_seed{s}']['test_calibrated'][k] for k in ['TP','TN','FP','FN','threshold']} for s in cfg['seeds']}
identities={x['sample_id']:x for x in rows(R/'inputs/test.jsonl')};paired=[]
for a in cfg['ablations']:
    for s in cfg['seeds']:
        f=f'full_seed{s}';b=f'{a}_seed{s}';fp={x['sample_id']:x for x in predictions[f]['test']}
        for x in predictions[b]['test']:
            y=fp[x['sample_id']];meta=identities[x['sample_id']];fullhat=int(y['probability']>allm[f]['threshold']);ablhat=int(x['probability']>allm[b]['threshold'])
            paired.append(dict(ablation=a,seed=s,sample_id=x['sample_id'],function_name=meta['function_name'],file_path=meta['file_path'],group_id=x['group_id'],label=x['label'],full_probability=y['probability'],ablation_probability=x['probability'],full_prediction=fullhat,ablation_prediction=ablhat,full_correct=fullhat==x['label'],ablation_correct=ablhat==x['label'],CE_full_minus_ablation=y['cross_entropy']-x['cross_entropy']))
with (R/'PAIRED_TEST_PREDICTIONS.jsonl').open('x',encoding='utf-8') as f:
    for x in paired:f.write(json.dumps(x,ensure_ascii=False)+'\n')
totals={k:0 for k in ['train_forwards','backwards','optimizer_steps','evaluation_forwards','oom']};params={};times=[];memory=[]
for name in cfg['runs']:
    v=read(S/'runs'/name/'summary.json');times.append(v['seconds']);memory.append(v['peak_allocated_bytes'])
    for k in totals:totals[k]+=v['counts'][k]
    params[v['arm']['ablation']]={k:v['initialization'][k] for k in ['installed_parameters','trainable_parameters','frozen_parameters']}
preflight={k:sum(read(S/f'PREFLIGHT_GPU{g}.json')['counts'][k] for g in [0,1]) for k in ['train_forwards','backwards','optimizer_steps','evaluation_forwards']}
preflight['initial_checkpoint_loads']=sum(len(read(S/f'PREFLIGHT_GPU{g}.json')['checks']) for g in [0,1])
params['full']={k:read(S/'PREFLIGHT_GPU1.json')['checks']['full_seed123']['initialization'][k] for k in ['installed_parameters','trainable_parameters','frozen_parameters']}
gradient_activity={}
for name in cfg['runs']:
    checks=read(S/'runs'/name/'gradient_audit.json')
    assert [x['step'] for x in checks]==[1,2,100,800]
    for x in checks:
        for branch,d in x['branches'].items():
            if 'trainable' in d and not d['trainable']:assert d['grad_tensors']==d['nonzero_grad_tensors']==0 and d['grad_norm_sq']==0
    gradient_activity[name]=checks
save(R/'GRADIENT_ACTIVITY.json',dict(frozen_branches_zero_gradient=True,checks=gradient_activity,scope='Four recorded steps per run; absent gradients on active subfeatures may reflect sample availability or zero-initialized heads, not automatically a disconnected implementation.'))
constant={}
for split in ['train','validation','test']:
    rec=rows(R/'inputs'/f'{split}.jsonl');constant[split]={}
    for y in [0,1]:
        pred=[dict(label=x['label'],probability=float(y),logits=[0.,0.]) for x in rec]
        m=recompute(pred,.5);m.pop('mean_CE');constant[split][str(y)]=m
summaryout=dict(status='COMPLETE',independent_metric_recomputations=verified,groups=aggregate,paired_effects_full_minus_ablation=effects,
                confusion_by_seed=counts,parameters=params,training_counts=totals,preflight_counts=preflight,
                final_evaluation_counts=read(E/'SUMMARY.json')['counts'],summed_training_seconds=sum(times),max_allocated_MiB=max(memory)/2**20,
                constant_predictors=constant,
                negative_recall_Wilson_intervals={str(s):wilson(allm[f'full_seed{s}']['test_calibrated']['TN'],6) for s in cfg['seeds']},
                limitations=['report warning labels, not vulnerability truth','historical exposed intra-project test','source file overlap','six test negatives','same weights across optimization seeds','input-mask ablations not active-capacity matched','no param binding examples in validation/test','only one guard-affected example per evaluation split'])
save(R/'ABLATION_SUMMARY.json',summaryout)
def mean_sd(a,split,m):
    x=aggregate[a][split][m];return f"{x['mean']:.4f} ± {x['sample_sd']:.4f}"
lines=['# XC2263N 消融实验完整结果表','', '数值为三个优化 seed 的均值 ± 样本标准差。三个 seed 共享未训练初值；不是三个独立初始化。历史测试不是本轮新盲测。','']
for split,kind in [('validation','calibrated'),('test','calibrated'),('test','fixed'),('train','calibrated')]:
    key=f'{split}_{kind}';lines += [f'## {split} / {kind}','','| 组别 | 正类F1 | macro-F1 | AUROC | AP | CE | 正类召回 | 负类召回 |','|---|---|---|---|---|---|---|---|']
    for a in groups:lines.append('| '+a+' | '+' | '.join(mean_sd(a,key,m) for m in ['F1','macro_F1','AUROC','AP','mean_CE','recall','negative_recall'])+' |')
    lines.append('')
lines+=['## 历史测试逐 seed（验证校准阈值）','','| 组别 | seed | 阈值 | TP/TN/FP/FN | macro-F1 | AUROC |','|---|---|---|---|---|---|']
for a in groups:
    for s in cfg['seeds']:
        m=allm[f'{a}_seed{s}']['test_calibrated'];cm='/'.join(str(m[k]) for k in ['TP','TN','FP','FN'])
        lines.append(f"| {a} | {s} | {m['threshold']:.8f} | {cm} | {m['macro_F1']:.4f} | {m['AUROC']:.4f} |")
lines+=['','## 配对差值：完整组 − 消融组（历史测试）','','正差值在F1/AUROC中倾向支持完整组；CE的负差值倾向支持完整组。不能将不同seed上的同一个函数当作新的独立样本。','','| 干预 | macro-F1差均值±SD | 三个seed差值 | AUROC差均值±SD | CE差均值±SD |','|---|---|---|---|---|']
for a in cfg['ablations']:
    e=effects[a]['test_calibrated'];fmt=lambda m:f"{e[m]['mean']:+.4f} ± {e[m]['sample_sd']:.4f}"
    lines.append('| '+a+' | '+fmt('macro_F1')+' | '+', '.join(f'{x:+.4f}' for x in e['macro_F1']['values'])+' | '+fmt('AUROC')+' | '+fmt('mean_CE')+' |')
(R/'ALL_ABLATION_TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(status='COMPLETE',groups=len(groups),metrics_verified=verified,training_counts=totals,full=aggregate['full']['test_calibrated']),ensure_ascii=False))
