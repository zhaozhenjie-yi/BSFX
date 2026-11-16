from ablation_runtime import *
import time,gc,traceback,datetime
start=time.time();counts=dict(evaluation_forwards=0,checkpoint_loads=0,initial_checkpoint_loads=0,fitted_checkpoint_loads=0,oom=0)
O=R/'final_evaluation';O.mkdir(exist_ok=False)
def predict_components(m,rec,data):
    m.eval();out=[]
    with torch.inference_mode():
        for r in rec:
            d=data[r['sample_id']];o=m(move(d['batch']),move(d['context']['v2']));counts['evaluation_forwards']+=1
            z=o['logits'][0].double();assert torch.isfinite(z).all();y=r['label']
            parts={k:float((o[v][0,1]-o[v][0,0]).double()) for k,v in [('base','base_logits'),('call','context_delta'),('semantic','semantic_delta')]}
            assert abs(sum(parts.values())-float(z[1]-z[0]))<1e-5
            out.append(dict(sample_id=r['sample_id'],group_id=r['group_id'],file_path=r['file_path'],function_sha256=r['function_sha256'],label=y,
                            probability=float(z.softmax(-1)[1]),logits=z.cpu().tolist(),cross_entropy=float(torch.logsumexp(z,0)-z[y]),
                            logit_margin_components=parts,component_scope='additive logit decomposition of this fitted arm; not an independent causal contribution'))
    return out
try:
    cfg=read(R/'EXPERIMENT_CONFIG.json');done=read(R/'TRAINING_COMPLETE.json')
    assert done['status']=='COMPLETE' and len(done['runs'])==len(cfg['runs'])
    assert not (R/'STOP.json').exists()
    assert all(x['exit_code']==0 for x in done['runs'])
    for p,h in read(R/'FROZEN_PAYLOAD.json')['files'].items():assert sha(R/p)==h,('payload_changed',p)
    save(O/'environment.json',setup())
    frozen={}
    for name,arm in cfg['runs'].items():
        d=R/'runs'/name;s=read(d/'summary.json');cal=read(d/'VALIDATION_CALIBRATION.json')
        assert s['status']=='COMPLETE' and s['counts']['optimizer_steps']==800
        assert sha(d/'final_checkpoint.pt')==s['checkpoint_sha256']
        vp=read(d/'validation_predictions.json');selected=threshold_choice(vp)
        assert selected==cal['selected']
        frozen[name]=dict(threshold=selected['threshold'],checkpoint_sha256=s['checkpoint_sha256'],validation_predictions_sha256=sha(d/'validation_predictions.json'))
    prior=read(PREV/'FINAL_SELECTION.json')['models']
    for item in prior:
        s=item['arm']['seed'];vp=read(PREV/'runs'/item['name']/'validation_step0800.json')
        assert abs(threshold_choice(vp)['threshold']-item['threshold'])<1e-12
        assert sha(PREV/item['checkpoint'])==item['checkpoint_sha256']
        frozen[f'full_seed{s}']=dict(threshold=item['threshold'],checkpoint_sha256=item['checkpoint_sha256'],reused=True)
    save(O/'FROZEN_THRESHOLDS.json',dict(created_at=datetime.datetime.now().astimezone().isoformat(),policy=cfg['threshold_policy'],models=frozen))
    # No test tensor has been loaded by this evaluator before freezing thresholds above.
    test,testdata=load_data('test');allresults={}
    def validate_prediction_identity(pred,split):
        truth={x['sample_id']:x for x in rows(R/'inputs'/f'{split}.jsonl')}
        assert len(pred)==len(truth)==len({x['sample_id'] for x in pred})
        for x in pred:
            y=truth[x['sample_id']]
            for k in ['label','group_id','file_path','function_sha256']:assert x[k]==y[k],(split,k,x['sample_id'])
            z=np.array(x['logits'],dtype=np.float64);p=float(np.exp(z[1]-np.logaddexp(z[0],z[1])))
            assert abs(p-x['probability'])<1e-12
    def result(name,arm,tp,vp,yp,reused):
        for pred,split in [(tp,'train'),(vp,'validation'),(yp,'test')]:validate_prediction_identity(pred,split)
        t=frozen[name]['threshold']
        return dict(arm=arm,reused=reused,threshold=t,
                    train_fixed=metrics(tp),train_calibrated=metrics(tp,t),
                    validation_fixed=metrics(vp),validation_calibrated=metrics(vp,t),
                    test_fixed=metrics(yp),test_calibrated=metrics(yp,t))
    for item in prior:
        name=item['name'];s=item['arm']['seed'];key=f'full_seed{s}'
        vp=read(PREV/'runs'/name/'validation_step0800.json')
        tp=read(PREV/'final_test'/f'{name}_training_predictions.json');yp=read(PREV/'final_test'/f'{name}_predictions.json')
        save(O/f'{key}_train.json',tp);save(O/f'{key}_validation.json',vp);save(O/f'{key}_test.json',yp)
        allresults[key]=result(key,dict(cfg['base'],ablation='full',seed=s),tp,vp,yp,True)
    interventions={}
    for name,arm in cfg['runs'].items():
        m,_=model(arm);counts['initial_checkpoint_loads']+=1;counts['checkpoint_loads']+=1
        cp=torch.load(R/'runs'/name/'final_checkpoint.pt',map_location='cpu',weights_only=True)
        counts['fitted_checkpoint_loads']+=1;counts['checkpoint_loads']+=1
        m.load_state_dict(cp['model_state_dict'],strict=True)
        interventions[name]=dict(input_mask=arm['ablation'],new_extractor=False,parameter_initialization_unchanged=True)
        m.cuda()
        yp=predict_components(m,test,testdata);save(O/f'{name}_test.json',yp)
        tp=read(R/'runs'/name/'training_predictions.json');vp=read(R/'runs'/name/'validation_predictions.json')
        allresults[name]=result(name,arm,tp,vp,yp,False)
        print(name,'COMPLETE',flush=True)
        del m,cp;gc.collect();torch.cuda.empty_cache()
    save(O/'ALL_METRICS.json',allresults)
    save(O/'MODEL_INTERVENTION_AUDIT.json',interventions)
    save(O/'SUMMARY.json',dict(status='COMPLETE',models=len(allresults),new_evaluations=len(cfg['runs']),reused_evaluations=3,counts=counts,seconds=time.time()-start,threshold_freeze_sha256=sha(O/'FROZEN_THRESHOLDS.json'),historical_test_not_blind=True))
except BaseException as exc:
    oom=isinstance(exc,torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower();counts['oom']+=int(oom)
    save(O/'FAILURE.json',dict(status='STOPPED_OOM' if oom else 'FAILED',error=repr(exc),traceback=traceback.format_exc(),counts=counts));traceback.print_exc();sys.exit(42 if oom else 1)
