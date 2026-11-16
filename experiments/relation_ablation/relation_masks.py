"""Input-only ablations of existing paths. No extractor or graph is added.

All returned dictionaries are shallow copies; replaced tensors are new tensors.
Historical cached tensors are never modified in place.
"""

ARMS = ('no_call_edges', 'no_param_binding', 'no_guard_input', 'no_ref_input')


def field(batch, name):
    for key in (name, 'packed_' + name):
        if key in batch:
            return batch[key]
    raise KeyError(name)


def mask_inputs(batch, context, arm):
    if arm == 'full':
        return batch, context
    assert arm in ARMS, arm
    b = dict(batch)
    c = dict(context)
    if arm == 'no_call_edges' and c['availability']['legacy']:
        legacy = dict(c['legacy'])
        legacy['call_mask'] = legacy['call_mask'] * 0
        c['legacy'] = legacy
    if arm in ('no_param_binding', 'no_guard_input') and c['availability']['v2']:
        s = dict(c['semantics'])
        if arm == 'no_param_binding':
            for key in ('binding_call', 'binding_actual_kind',
                        'binding_formal_kind', 'binding_numeric'):
                s[key] = s[key][:0]
            s['counts'] = s['counts'].clone()
            s['counts'][1] = 0
        else:
            s['guard_indices'] = s['guard_indices'] * 0 - 1
            # This bit is computed from constant-false guards in the extractor.
            # Remove it as well; retain general partial/uncertain-flow metadata.
            s['event_numeric'] = s['event_numeric'].clone()
            s['event_numeric'][:, 10] = 0
        c['semantics'] = s
    if arm == 'no_ref_input':
        for name in ('ref_identity_ids', 'ref_definition_token_ids', 'ref_link_matrix'):
            field(b, name)  # fail explicitly if the runtime field was renamed
            for key in (name, 'packed_' + name):
                if key in b:
                    b[key] = b[key] * 0
    return b, c


def affected_counts(batch, context):
    """Raw counts, not learned scores; can run before GPU/model loading."""
    result = {a: 0 for a in ARMS}
    if context['availability']['legacy']:
        result['no_call_edges'] = int(context['legacy']['call_mask'].sum())
    if context['availability']['v2']:
        s = context['semantics']
        result['no_param_binding'] = len(s['binding_call'])
        result['no_guard_input'] = int((s['guard_indices'] >= 0).sum())
        result['no_guard_input'] += int((s['event_numeric'][:, 10] != 0).sum())
    result['no_ref_input'] = int((field(batch, 'ref_identity_ids') != 0).sum())
    result['no_ref_input'] += int((field(batch, 'ref_definition_token_ids') != 0).sum())
    result['no_ref_input'] += int((field(batch, 'ref_link_matrix') != 0).sum())
    return result
