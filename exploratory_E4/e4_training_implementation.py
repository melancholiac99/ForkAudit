"""Audit the narrowly authorized checkpoint optimization against the original frozen source."""
from pathlib import Path
import hashlib
import json

TRAINING = 'exploratory_E4/e4_06_train.py'
MANIFEST = 'exploratory_E4/run/training_implementation_change.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def expected_optimized_source(original):
    """Only two execution changes, their assertion, documentation and provenance fields."""
    replacements = [
        ('(micro-batch 1, gradient accumulation); 2 epochs; bf16 base weights, fp32 LoRA weights; gradient checkpointing.',
         '(micro-batch 1, gradient accumulation); 2 epochs; bf16 base weights, fp32 LoRA weights; gradient checkpointing disabled.'),
        ('    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})',
         '    model.gradient_checkpointing_disable()'),
        ('    def seq_logp(pi, ri):',
         '    assert not model.is_gradient_checkpointing\n'
         '    training_implementation = dict(\n'
         '        name="no_checkpoint_v1", source_sha256=E.file_sha256(__file__),\n'
         '        gradient_checkpointing=bool(model.is_gradient_checkpointing), training_use_cache=False,\n'
         '        reference_use_cache="model_default", vocabulary_projection="full",\n'
         '        authorization_manifest_sha256=E.file_sha256(os.path.join(E.HERE, "run", "training_implementation_change.json")))\n'
         '    print("TRAINING_IMPLEMENTATION " + json.dumps(training_implementation), flush=True)\n\n'
         '    def seq_logp(pi, ri):'),
        ('            logits = model(input_ids=ids).logits[0, len(pi) - 1:-1].float()',
         '            logits = model(input_ids=ids, use_cache=False if model.training else None).logits[0, len(pi) - 1:-1].float()'),
        ('        pairs_sha256=E.file_sha256(os.path.join(E.DATA, "pairs", f"{args.arm}.jsonl")), model=args.model))',
         '        pairs_sha256=E.file_sha256(os.path.join(E.DATA, "pairs", f"{args.arm}.jsonl")), model=args.model,\n'
         '        training_implementation=training_implementation))'),
    ]
    for before, after in replacements:
        assert original.count(before) == 1, before
        original = original.replace(before, after, 1)
    return original


def verify_protected_inputs(root):
    root = Path(root)
    frozen = read(root/'exploratory_E4/run/protected_inputs.json')
    change = read(root/MANIFEST) if (root/MANIFEST).exists() else None
    authorized = {}
    for name, digest in frozen.items():
        actual = sha(root/name)
        if actual == digest:
            continue
        assert name == TRAINING and change is not None, name
        assert change['optimization'] == 'disable_gradient_checkpointing_only'
        assert change['original_sha256'] == digest
        assert change['optimized_sha256'] == actual
        assert sha(root/change['original_snapshot']) == digest
        assert expected_optimized_source((root/change['original_snapshot']).read_text()) == (root/name).read_text()
        assert change['verifier_sha256'] == sha(Path(__file__))
        control_path = root/change['numerical_control']
        assert sha(control_path) == change['numerical_control_sha256']
        controls = read(control_path)['summary']
        for variant in ['original', 'no_checkpoint']:
            assert controls[variant]['all_passed'] and controls[variant]['all_bitwise_equal']
            assert controls[variant]['max_loss_difference'] == controls[variant]['max_gradient_relative_l2'] == 0
        for path, expected in change['preserved_legacy_artifacts'].items():
            assert sha(root/path) == expected, path
        authorized[name] = dict(original_sha256=digest, current_sha256=actual, manifest=MANIFEST)
    if change is not None:
        assert TRAINING in authorized, 'Declared optimization must be present'
    return dict(protected_files_unchanged=len(frozen)-len(authorized), authorized_changes=authorized)


def verify_training_log(root, tag, log):
    root = Path(root)
    if not (root/MANIFEST).exists():
        return
    change = read(root/MANIFEST)
    if tag in change['legacy_training_tags']:
        assert 'training_implementation' not in log, tag
        return
    impl = log['training_implementation']
    assert impl['name'] == 'no_checkpoint_v1'
    assert impl['source_sha256'] == change['optimized_sha256']
    assert impl['authorization_manifest_sha256'] == sha(root/MANIFEST)
    assert impl['gradient_checkpointing'] is False and impl['training_use_cache'] is False
    assert impl['reference_use_cache'] == 'model_default' and impl['vocabulary_projection'] == 'full'
