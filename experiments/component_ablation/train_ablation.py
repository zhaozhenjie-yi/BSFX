from ablation_runtime import *
import time, traceback
name=sys.argv[1];cfg=read(R/'EXPERIMENT_CONFIG.json');arm=cfg['runs'][name]
O=R/'runs'/name;O.mkdir(parents=True,exist_ok=False);start=time.time()
counts=dict(train_forwards=0,backwards=0,optimizer_steps=0,evaluation_forwards=0,oom=0,checkpoint_loads=0)
def progress(stage,step):
    p=O/'progress.tmp';p.write_text(json.dumps(dict(stage=stage,step=step,seconds=time.time()-start,counts=counts)));p.replace(O/'progress.json')
try:
    assert read(R/'PREFLIGHT_COMPLETE.json')['status']=='PASS'
    assert not (R/'STOP.json').exists()
    save(O/'environment.json',setup())
    tr,td=load_data('train');va,vd=load_data('validation');byid={r['sample_id']:r for r in tr}
    m,info=model(arm);counts['checkpoint_loads']+=1
    assert info['shared_initial_digest_before_intervention']==cfg['initial_digest']
    save(O/'initialization.json',info);m.cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=arm['lr'],weight_decay=.01,betas=(.9,.999),eps=1e-8)
    assert not opt.state
    seed(arm['seed']);weights=torch.tensor([225/(2*35),225/(2*190)],device='cuda')
    trace=(O/'training_trace.jsonl').open('x');grads=[];seen={}
    for row in schedule(arm['seed']):
        step=row['step'];assert not (R/'STOP.json').exists(),'OTHER_RUN_FAILED_STOP'
        assert time.time()-start<cfg['max_seconds_per_run'],'TIME_BUDGET_EXHAUSTED'
        progress('training',step);m.train();opt.zero_grad(set_to_none=True);ce_total=0.;obj=0.
        for sid,y in zip(row['sample_ids'],row['labels']):
            r=byid[sid];assert y==r['label'];d=td[sid]
            b=move(d['batch']);c=move(d['context']['v2']);o=m(b,c);counts['train_forwards']+=1
            ce=torch.nn.functional.cross_entropy(o['logits'],torch.tensor([y],device='cuda'),reduction='none')[0]
            loss=ce*(weights[y] if arm['loss']=='weighted_ce' else 1.)/2
            assert torch.isfinite(loss);loss.backward();counts['backwards']+=1
            ce_total+=float(ce.detach())/2;obj+=float(loss.detach());seen[sid]=seen.get(sid,0)+1
            del b,c,o,ce,loss
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm)
        if step in [1,2,100,800]:grads.append(dict(step=step,branches=gradient_audit(m)))
        opt.step();counts['optimizer_steps']+=1
        trace.write(json.dumps(dict(step=step,sample_ids=row['sample_ids'],labels=row['labels'],mean_CE=ce_total,objective=obj,gradient_norm=float(norm)))+'\n');trace.flush()
        if step%100==0:print(name,step,'CE',ce_total,flush=True)
    trace.close()
    progress('validation',800)
    vp=predict(m,va,vd,arm,counts);save(O/'validation_predictions.json',vp)
    cal=threshold_choice(vp);save(O/'VALIDATION_CALIBRATION.json',dict(policy='prior_fixed_midpoint_policy',selected=cal))
    tp=predict(m,tr,td,arm,counts);save(O/'training_predictions.json',tp)
    cp=O/'final_checkpoint.pt';torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in m.state_dict().items()},arm=arm,optimizer_step=800,initial_digest=info['shared_initial_digest_before_intervention']),cp)
    save(O/'gradient_audit.json',grads);save(O/'training_exposure.json',seen)
    result=dict(status='COMPLETE',arm=arm,counts=counts,initialization=info,seconds=time.time()-start,
                checkpoint_sha256=sha(cp),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                train_fixed=metrics(tp),train_calibrated=metrics(tp,cal['threshold']),
                validation_fixed=metrics(vp),validation_calibrated=cal,test_evaluated=False)
    save(O/'summary.json',result);progress('complete',800);print(json.dumps(result),flush=True)
except BaseException as exc:
    oom=isinstance(exc,torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower()
    counts['oom']+=int(oom)
    fail=dict(status='STOPPED_OOM' if oom else 'FAILED',error=repr(exc),traceback=traceback.format_exc(),counts=counts)
    save(O/'failure.json',fail)
    try:save(R/'STOP.json',dict(run=name,**fail))
    except FileExistsError:pass
    progress(fail['status'],counts['optimizer_steps']);traceback.print_exc();sys.exit(42 if oom else 1)
