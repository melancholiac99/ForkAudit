"""CPU execution check of the actual archived/new seq_logp functions and optimizer step."""
from pathlib import Path
import ast
import copy
import json
import sys
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'exploratory_E4'))
from e4_training_implementation import expected_optimized_source, verify_protected_inputs, verify_training_log, sha


def extract(path, model):
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    seq = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'seq_logp')
    class CPU(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant('cpu'), node) if node.value == 'cuda' else node
    # Only move device/autocast literals to CPU; execute all actual indexing/cache logic.
    seq = CPU().visit(seq)
    module = ast.fix_missing_locations(ast.Module(body=[seq], type_ignores=[]))
    scope = dict(torch=torch, model=model)
    exec(compile(module, str(path), 'exec'), scope)
    return scope['seq_logp']


def main():
    torch.set_num_threads(1); torch.manual_seed(9021)
    old = ROOT/'exploratory_E4/run/frozen_training/e4_06_train_original.py'
    new = ROOT/'exploratory_E4/e4_06_train.py'
    expected = expected_optimized_source(old.read_text())
    assert expected == new.read_text()
    assert expected != new.read_text().replace('BETA, R, ALPHA, DROPOUT, EFF_BATCH, EPOCHS, MAX_LEN = 0.1',
                                               'BETA, R, ALPHA, DROPOUT, EFF_BATCH, EPOCHS, MAX_LEN = 0.2')
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)
    cfg._attn_implementation = 'sdpa'
    model = Qwen2ForCausalLM(cfg).to(dtype=torch.bfloat16)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                           target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                                           bias='none', task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
                if 'lora_B' in name:
                    p.normal_(0, .01)
    state = copy.deepcopy(model.state_dict())
    functions = [extract(old, model), extract(new, model)]
    reports = []
    for kind in ['DPO', 'SFT']:
        outputs = []
        for index, seq in enumerate(functions):
            model.load_state_dict(state); model.zero_grad(set_to_none=True)
            if index == 0:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            else:
                model.gradient_checkpointing_disable()
            pi = list(range(1, 20)); chosen = [20, 21, 22]; rejected = [23, 24, 25, 26, 27]
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                refs = [float(seq(pi, response)[0]) for response in [chosen, rejected]]
            model.train(); torch.manual_seed(173)
            if kind == 'DPO':
                pc, pr = seq(pi, chosen)[0], seq(pi, rejected)[0]
                loss = -torch.nn.functional.logsigmoid(.1*((pc-refs[0])-(pr-refs[1])))
            else:
                loss = -seq(pi, chosen)[1]
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=5e-6, weight_decay=0.)
            (loss/16).backward()
            gradients = torch.cat([p.grad.flatten().clone() for p in params])
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step()
            updated = torch.cat([p.detach().flatten().clone() for p in params])
            outputs.append((refs, float(loss.detach()), gradients, updated))
        a, b = outputs
        assert a[0] == b[0] and a[1] == b[1]
        assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
        reports.append(dict(kind=kind, reference_equal=True, loss_equal=True, gradients_equal=True, optimizer_update_equal=True))
    assert not torch.cuda.is_initialized()
    protection = verify_protected_inputs(ROOT)
    change = json.loads((ROOT/'exploratory_E4/run/training_implementation_change.json').read_text())
    implementation = dict(name='no_checkpoint_v1',source_sha256=change['optimized_sha256'],
                          authorization_manifest_sha256=sha(ROOT/'exploratory_E4/run/training_implementation_change.json'),
                          gradient_checkpointing=False,training_use_cache=False,reference_use_cache='model_default',vocabulary_projection='full')
    verify_training_log(ROOT,'A_s1',dict(training_implementation=implementation))
    try:
        verify_training_log(ROOT,'A_s1',dict(training_implementation=dict(implementation, gradient_checkpointing=True)))
    except AssertionError:
        pass
    else:
        raise AssertionError('Runtime implementation audit failed to reject incorrect checkpoint mode')
    report=dict(passed=True, scope='Actual production seq_logp bodies, CUDA literals adapted to CPU; tiny model only',
                optimized_sha256=sha(new),checks=reports,protection=protection,audit_rejects_wrong_runtime_mode=True)
    (HERE/'deployment_tests.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
