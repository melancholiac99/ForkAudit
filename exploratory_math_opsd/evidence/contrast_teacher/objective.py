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
        raw = 0.5 * (plus - minus)
        log_target = F.log_softmax(lp.detach() + raw.clamp(-3, 3), dim=-1)
        log_target = torch.where(has_negative[..., None], log_target, lp.detach())
        clipped_mass = (log_target.exp() * (raw.abs()>3)).sum(-1) * has_negative
    return has_negative * F.kl_div(lp, log_target, log_target=True, reduction='none').sum(-1), log_target, clipped_mass
