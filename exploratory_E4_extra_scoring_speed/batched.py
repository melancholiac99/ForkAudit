"""Exploratory candidate batching; never imported by the formal E4 scorer."""
import copy


def score_batched(scorer, job, batch_size=5):
    t, m = scorer.t, scorer.model
    device = next(m.parameters()).device
    pre = job['prefix_ids']
    cmds = [c['suffix_ids'] for c in job['candidates']]
    assert pre and all(len(cmd) >= 2 for cmd in cmds)
    actual_batches = []
    with t.inference_mode():
        out = m.model(input_ids=t.tensor([pre], device=device), use_cache=True)
        lp0 = t.log_softmax(m.lm_head(out.last_hidden_state[0, -1].float()), -1)
        cache = out.past_key_values
        del out
        cache_bytes = sum(layer.keys.numel() * layer.keys.element_size()
                          + layer.values.numel() * layer.values.element_size()
                          for layer in cache.layers)
        rows = []
        offset = 0
        while offset < len(cmds):
            width = min(batch_size, len(cmds) - offset)
            if device.type == 'cuda':
                free, _ = t.cuda.mem_get_info(device)
                # Account conservatively for copy/repeat transient allocations and activations.
                while width > 1 and cache_bytes * (width + 2) + 2 * 1024**3 > free:
                    width -= 1
            group = cmds[offset:offset + width]
            longest = max(len(cmd) - 1 for cmd in group)
            ids = t.zeros((width, longest), dtype=t.long, device=device)
            mask = t.zeros((width, len(pre) + longest), dtype=t.long, device=device)
            mask[:, :len(pre)] = 1
            for i, cmd in enumerate(group):
                ids[i, :len(cmd) - 1] = t.tensor(cmd[:-1], device=device)
                mask[i, len(pre):len(pre) + len(cmd) - 1] = 1
            kv = copy.deepcopy(cache)
            kv.batch_repeat_interleave(width)
            positions = t.arange(len(pre), len(pre) + longest, device=device)[None, :].expand(width, -1)
            hidden = m.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                             past_key_values=kv, use_cache=False).last_hidden_state
            # Only project the real suffix positions; preserve FP32 vocabulary projection.
            lengths = [len(cmd) - 1 for cmd in group]
            packed = t.cat([hidden[i, :n] for i, n in enumerate(lengths)], dim=0)
            lp = t.log_softmax(m.lm_head(packed.float()), -1)
            targets = t.tensor([token for cmd in group for token in cmd[1:]], device=device)
            selected = lp[t.arange(len(targets), device=device), targets].tolist()
            first = lp0[t.tensor([cmd[0] for cmd in group], device=device)].tolist()
            begin = 0
            for i, n in enumerate(lengths):
                rows.append([first[i]] + selected[begin:begin + n])
                begin += n
            actual_batches.append(width)
            offset += width
            del kv, hidden, packed, lp, ids, mask, positions, targets
    return rows, actual_batches
