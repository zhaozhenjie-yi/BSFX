from runtime import *
import time,traceback,math
name=sys.argv[1];cfg=read(R/'RUN_CONFIG.json');arm=cfg['arms'][name]
O=R/'runs'/name;O.mkdir(parents=True,exist_ok=False);started=time.time()
counts=dict(initial_checkpoint_loads=0,train_forwards=0,backwards=0,optimizer_steps=0,evaluation_forwards=0,oom=0)
progress={};best=None
def update(**kw):
 progress.update(kw);p=O/'progress.tmp';p.write_text(json.dumps(dict(progress=progress,counts=counts,seconds=time.time()-started)));p.replace(O/'progress.json')
def atomic_checkpoint(path,value):
 tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
try:
 assert read(R/'PREPARATION_RESULT.json')['status']=='PASS'
 assert not list((R/'runs').glob('*/OOM.json')),'EARLIER_OOM_STOP'
 env=setup();save(O/'environment.json',env)
 train,trdata=load_data('train');valid,vdata=load_data('validation')
 assert len(train)==225 and len(valid)==54 and not {x['sample_id'] for x in train}&{x['sample_id'] for x in valid}
 # Test records and test tensors are deliberately not loaded during fitting or selection.
 m=model(arm);counts['initial_checkpoint_loads']+=1;initial=digest(m);m.cuda()
 optimizer=torch.optim.AdamW(m.parameters(),lr=arm['lr'],weight_decay=.01,betas=(.9,.999),eps=1e-8)
 assert all(p.requires_grad for p in m.parameters())
 save(O/'initialization.json',dict(arm=arm,full_model_digest=initial,trainable_parameters=sum(p.numel() for p in m.parameters()),optimizer=dict(type='AdamW',lr=arm['lr'],weight_decay=.01),all_parameters_trainable=True,empty_optimizer_state=not bool(optimizer.state)))
 seed(arm['seed']);rng=random.Random(arm['seed']);bylabel={k:[x for x in train if x['label']==k] for k in [0,1]};queue=[]
 def sample():
  global queue
  if not queue:queue=train.copy();rng.shuffle(queue)
  return queue.pop()
 weights=torch.tensor([len(train)/(2*len(bylabel[k])) for k in [0,1]],device='cuda')
 p0=predict(m,valid,vdata,arm,counts);save(O/'validation_step0000.json',p0)
 save(O/'baseline_validation_metrics.json',dict(fixed=metrics(p0),tuned=threshold_choice(p0)))
 trace=(O/'training_trace.jsonl').open('x');seen=Counter() if False else {}
 grads=[];history=[]
 for step in range(1,arm['steps']+1):
  assert time.time()-started<cfg['max_seconds_per_arm'],'WALL_TIME_BUDGET_EXHAUSTED'
  update(stage='training',step=step);m.train();optimizer.zero_grad(set_to_none=True)
  pair=[rng.choice(bylabel[0]),rng.choice(bylabel[1])] if arm['sampling']=='balanced' else [sample(),sample()]
  ce_sum=0.;objective=0.
  for r in pair:
   sid=r['sample_id'];seen[sid]=seen.get(sid,0)+1;d=trdata[sid];b=move(d['batch']);c=move(d['context'])
   o=m(b,c if arm['route']=='v3' else c['v2']);counts['train_forwards']+=1
   target=torch.tensor([r['label']],device='cuda');ce=torch.nn.functional.cross_entropy(o['logits'],target,reduction='none')[0]
   # Mean reduction with class weights cancels for physical batch size 1. Apply the weight to unreduced CE instead.
   loss=ce*(weights[r['label']] if arm['loss']=='weighted_ce' else 1.)/2.
   assert torch.isfinite(loss),'NONFINITE_LOSS';loss.backward();counts['backwards']+=1
   ce_sum+=float(ce.detach())/2;objective+=float(loss.detach());del b,c,o,loss,ce
  norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);assert torch.isfinite(norm)
  if step in [1,2,100,arm['steps']]:
   branches={}
   for n,p in m.named_parameters():
    if p.grad is not None:
     branch=('v3' if n.startswith('v3.') else 'semantic_v2' if '.semantic.' in '.'+n else 'call_context' if '.context.' in '.'+n else 'base')
     q=branches.setdefault(branch,dict(tensors=0,nonzero_tensors=0,l2_sq=0.));q['tensors']+=1;q['nonzero_tensors']+=int(bool(p.grad.ne(0).any()));q['l2_sq']+=float(p.grad.double().square().sum())
   grads.append(dict(step=step,branches=branches))
  optimizer.step();counts['optimizer_steps']+=1
  trace.write(json.dumps(dict(step=step,sample_ids=[r['sample_id'] for r in pair],labels=[r['label'] for r in pair],mean_CE=ce_sum,objective=objective,gradient_norm=float(norm)))+'\n');trace.flush()
  if step in arm['eval_steps']:
   pred=predict(m,valid,vdata,arm,counts);fixed=metrics(pred);tuned=threshold_choice(pred)
   save(O/f'validation_step{step:04d}.json',pred);record=dict(step=step,fixed=fixed,tuned=tuned);history.append(record)
   key=(tuned['macro_F1'],tuned['balanced_accuracy'],-tuned['mean_CE'],-step)
   if best is None or key>best[0]:
    best=(key,record)
    atomic_checkpoint(O/'best_checkpoint.pt',dict(model_state_dict={k:v.detach().cpu() for k,v in m.state_dict().items()},optimizer_step=step,arm=arm,initial_digest=initial))
   save(O/f'metrics_step{step:04d}.json',record)
   print(json.dumps(dict(arm=name,step=step,validation=tuned,counts=counts)),flush=True)
  elif step%50==0:print(name,'step',step,'CE',round(ce_sum,5),flush=True)
  update(completed_steps=step)
 trace.close()
 # Report final fit separately from the chosen validation checkpoint.
 train_pred=predict(m,train,trdata,arm,counts);save(O/'training_final_predictions.json',train_pred)
 atomic_checkpoint(O/'final_checkpoint.pt',dict(model_state_dict={k:v.detach().cpu() for k,v in m.state_dict().items()},optimizer_step=arm['steps'],arm=arm,initial_digest=initial))
 save(O/'gradient_audit.json',grads);save(O/'training_exposure.json',seen)
 result=dict(status='COMPLETE',arm=arm,counts=counts,validation_history=history,selected=best[1],training_final=metrics(train_pred),training_final_selected_threshold=metrics(train_pred,best[1]['tuned']['threshold']),best_checkpoint_sha256=sha(O/'best_checkpoint.pt'),initial_digest=initial,seconds=time.time()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),unique_training_samples_seen=len(seen),test_evaluated=False)
 save(O/'summary.json',result);update(stage='complete');print(json.dumps(dict(status='COMPLETE',arm=name,seconds=result['seconds'],selected=best[1])),flush=True)
except Exception as e:
 oom=isinstance(e,torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower();counts['oom']+=int(oom)
 failure=dict(status='STOPPED_OOM' if oom else 'FAILED',error=repr(e),traceback=traceback.format_exc(),counts=counts,progress=progress,automatic_retry=False)
 save(O/('OOM.json' if oom else 'failure.json'),failure);update(stage=failure['status']);traceback.print_exc();sys.exit(42 if oom else 1)
