from runtime import *
import collections,time,traceback
O=R/'cache';O.mkdir(exist_ok=False);start=time.time()
try:
 allrows=[r for s in ['train','validation','test'] for r in rows(R/'inputs'/f'{s}.jsonl')]
 assert len(allrows)==310 and len({r['sample_id'] for r in allrows})==310
 manifest=read(R/'SPLIT_MANIFEST.json')
 for s,entry in manifest['partitions'].items():assert sha(R/entry['path'])==entry['sha256']
 F=E/'automotive_preprocess_repair_20261023_r1'
 existing={r['sample_id']:r for r in read(F/'preparation/eligible_manifest.json')}
 info={r['sample_id']:r for r in read(H/'preparation04/context_index.json')}
 vocab_path=P/'historical/validation230/evidence/vocab_tokens.json';vocab=read(vocab_path);vb=read(P/'historical/validation230/vocabulary_binding.json')
 assert sha(vocab_path)==vb['sha256'] and len(vocab)==9279
 class Adapter(bridge.R6ERepresentationAdapter):
  def __init__(self):
   self.dataset='xc2263n';self.representation='v2';self.labels={r['sample_id']:-1 for r in allrows}
   self.ordered_vocab_tokens=tuple(vocab);self.vocab_entry=vb['entry'];self.vocab=bridge.Vocabulary.from_ordered_tokens(vocab,special_tokens=['[PAD]','[UNK]','[SEG_START]','[SEG_END]'])
   self.shard_index=bridge.R6CShardIndex(F/'preparation','xc2263n');self.segment_index=None;self.parent_sidecar=None
 adapter=Adapter();audits=[]
 old={r['sample_id']:r for r in rows(H/'evaluation_v2_scope04/input_audit.jsonl')}
 keys=['packed_token_ids','packed_depth_ids','packed_sibling_order_ids','packed_global_position_ids','packed_local_position_ids','packed_delta_seg','packed_ref_identity_ids','packed_ref_definition_token_ids','packed_padding_mask','packed_sibling_parent_ids','packed_ast_node_identity_ids','packed_segment_ids','packed_special_token_ids','packed_ref_link_matrix','packed_path_to_sample','packed_path_slot']
 for r in allrows:
  sid=r['sample_id'];assert r['function_sha256']==existing[sid]['function_sha256']==info[sid]['function_sha256']
  b=adapter.collate([sid]);c=historic.prepared(sid,info[sid]);mask=b.get('attention_mask',b.get('packed_attention_mask'))
  consumed=dict({k:b[k] for k in keys},attention_mask=mask,batch_size=1)
  assert tree_digest(consumed)==old[sid]['local_input_sha256']
  assert tree_digest(c)==old[sid]['complete_context_sha256']
  # Only numeric model inputs. Metadata, report findings and ground-truth labels are never cached as features.
  batch={k:b[k] for k in keys};batch['attention_mask']=mask;batch['labels']=torch.tensor([-1],dtype=torch.long)
  torch.save(dict(batch=batch,context=c),O/(sid+'.pt'))
  tokens=batch['packed_token_ids'][~batch['packed_padding_mask']]
  audits.append(dict(sample_id=sid,split=r['split'],function_sha256=r['function_sha256'],cache_sha256=sha(O/(sid+'.pt')),local_input_sha256=old[sid]['local_input_sha256'],context_sha256=old[sid]['complete_context_sha256'],availability=info[sid]['availability'],valid_tokens=len(tokens),unk_tokens=int((tokens==1).sum()),paths=int(batch['packed_token_ids'].shape[0])))
 save(R/'CACHE_AUDIT.json',audits)
 imported={}
 for mod in list(sys.modules.values()):
  p=getattr(mod,'__file__',None)
  if p and str(E) in p and Path(p).suffix=='.py':imported[p]=sha(p)
 save(R/'RUNTIME_BINDINGS.json',dict(imported_modules=imported,vocabulary=dict(path=str(vocab_path),sha256=sha(vocab_path)),initial_state=dict(path=str(E/'context_v2_learning_ablation_20261024_r1/common_initial_state.pt'),sha256=sha(E/'context_v2_learning_ablation_20261024_r1/common_initial_state.pt'))))
 save(R/'PREPARATION_RESULT.json',dict(status='PASS',samples=310,forwards=0,training_steps=0,checkpoints_loaded=0,seconds=time.time()-start,max_paths=max(a['paths'] for a in audits),availability={k:sum(a['availability'][k] for a in audits) for k in ['legacy','v2','v3']},unk_fraction=sum(a['unk_tokens'] for a in audits)/sum(a['valid_tokens'] for a in audits),source_and_cached_inputs_match_archived_evaluation=True))
 print(json.dumps(read(R/'PREPARATION_RESULT.json')),flush=True)
except Exception as e:
 save(R/'PREPARATION_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
