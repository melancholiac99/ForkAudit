"""Contrastive target and unreduced KL for a batch of supervised positions."""
import torch
import torch.nn.functional as F


def contrast_kl(student, positive, negative, has_negative, temperature=1.1):
    # Keep double precision for the numerical checks; promote training logits to FP32.
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    lp = F.log_softmax(student.to(dtype) / temperature, dim=-1)
    with torch.no_grad():
        plus = F.log_softmax(positive.to(dtype) / temperature, dim=-1)
        minus = F.log_softmax(negative.to(dtype) / temperature, dim=-1)
        log_target = F.log_softmax(lp.detach() + 0.5 * (plus - minus), dim=-1)
        log_target = torch.where(has_negative[..., None], log_target, plus)
    return F.kl_div(lp, log_target, log_target=True, reduction='none').sum(-1), log_target
