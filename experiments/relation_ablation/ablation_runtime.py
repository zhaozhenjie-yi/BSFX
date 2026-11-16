"""Same historical model and training protocol, with scoped input masks only."""
from pathlib import Path
import sys, os, json, hashlib, importlib.util, types
R = Path(__file__).resolve().parent
PREV = Path('BSFX_WORKSPACE/experiments/xc2263n_report_training_20261110_r1')
spec = importlib.util.spec_from_file_location('previous_runtime', PREV / 'runtime.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
# previous_runtime inserts historical directories; use the already loaded local file explicitly.
mask_spec = importlib.util.spec_from_file_location('relation_masks', R / 'relation_masks.py')
masks = importlib.util.module_from_spec(mask_spec)
mask_spec.loader.exec_module(masks)
torch, np = old.torch, old.np
read, rows, sha, save = old.read, old.rows, old.sha, old.save
seed, setup, move, metrics, predict = old.seed, old.setup, old.move, old.metrics, old.predict


def baseline_name(s):
    return 'v2_weighted_natural_800' if s == 123 else f'confirm_v2_weighted_natural_800_seed{s}'


def load_data(split):
    assert sha(R / 'inputs' / f'{split}.jsonl') == sha(PREV / 'inputs' / f'{split}.jsonl')
    return old.load_data(split)


def threshold_choice(pred):
    p = sorted(set(x['probability'] for x in pred))
    ts = sorted(set([0., .5, 1.] + [(a + b) / 2 for a, b in zip(p, p[1:])]))
    return max((metrics(pred, t) for t in ts), key=lambda x: (
        x['macro_F1'], x['balanced_accuracy'], -abs(x['threshold'] - .5)))


def masked_forward(self, batch, context):
    b, c = masks.mask_inputs(batch, context, self._relation_arm)
    return self._unmasked_forward(b, c)


def model(arm):
    m = old.model(arm)
    initial = old.digest(m)
    a = arm['ablation']
    assert a == 'full' or a in masks.ARMS
    if a != 'full':
        m._relation_arm = a
        m._unmasked_forward = m.forward
        m.forward = types.MethodType(masked_forward, m)
    assert old.digest(m) == initial
    info = dict(shared_initial_digest_before_intervention=initial,
                post_intervention_digest=initial, disabled_parameters=[],
                installed_parameters=sum(p.numel() for p in m.parameters()),
                trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in m.parameters() if not p.requires_grad),
                intervention=a, intervention_type='input_mask_only',
                active_capacity_matched=False)
    return m, info


def gradient_audit(m):
    result = {}
    for n, p in m.named_parameters():
        k = '.'.join(n.split('.')[:2]) if n.startswith('base.') else n.split('.')[0]
        v = result.setdefault(k, dict(parameters=0, trainable=0, grad_tensors=0,
                                     nonzero_grad_tensors=0, grad_norm_sq=0.))
        v['parameters'] += p.numel()
        v['trainable'] += p.numel() * int(p.requires_grad)
        if p.grad is not None:
            assert p.requires_grad
            v['grad_tensors'] += 1
            v['nonzero_grad_tensors'] += int(bool(p.grad.ne(0).any()))
            v['grad_norm_sq'] += float(p.grad.double().square().sum())
    watched = {'call_edge_input': m.context.edge_input,
               'call_message_in': m.context.message_in,
               'call_message_out': m.context.message_out,
               'param_binding': m.semantic.binding,
               'hpe_ref_identity': m.base.hpe.ref_identity_embedding,
               'hpe_ref_definition': m.base.hpe.ref_definition_projection}
    for key, module in watched.items():
        pp = list(module.parameters())
        result['relation_path.' + key] = dict(
            parameters=sum(p.numel() for p in pp),
            grad_tensors=sum(p.grad is not None for p in pp),
            nonzero_grad_tensors=sum(p.grad is not None and bool(p.grad.ne(0).any()) for p in pp),
            grad_norm_sq=sum(float(p.grad.double().square().sum()) for p in pp if p.grad is not None))
    grad = m.semantic.event[0].weight.grad
    result['relation_path.guard_event_columns'] = dict(
        grad_present=grad is not None,
        guard_grad_norm_sq=float(grad[:, :96].double().square().sum()) if grad is not None else None,
        dead_flag_grad_norm_sq=float(grad[:, 318].double().square().sum()) if grad is not None else None)
    return result


def schedule(s):
    p = PREV / 'runs' / baseline_name(s) / 'training_trace.jsonl'
    result = rows(p)
    assert len(result) == 800
    assert sha(p) == read(R / 'EXPERIMENT_CONFIG.json')['schedule_sha256'][str(s)]
    return result
