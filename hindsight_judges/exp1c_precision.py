"""Explicit 1c precision recovery; original checkpoint and P4 backend stay intact."""

POLICY = "fp16_transformer_fp32_head_preserved_buffers"


def set_flags(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def configure(frozen):
    """Cast transformer parameters only. Module.half() would also round RoPE."""
    t, model = frozen.t, frozen.model
    assert all(p.dtype == t.bfloat16 for p in model.model.parameters())
    assert model.lm_head.weight.dtype == t.float32
    buffers = {name: value.clone() for name, value in model.model.named_buffers()}
    for param in model.model.parameters():
        param.data = param.data.to(dtype=t.float16)
    assert all(p.dtype == t.float16 and t.isfinite(p).all() for p in model.model.parameters())
    assert all(value.dtype == buffers[name].dtype and t.equal(value, buffers[name])
               for name, value in model.model.named_buffers()), "position/other buffers changed"
    assert model.lm_head.weight.dtype == t.float32
    return dict(policy=POLICY, transformer_dtype="torch.float16", head_dtype="torch.float32",
                buffers_unchanged=True, buffer_dtypes={name: str(b.dtype) for name, b in buffers.items()})
