"""§18 content teacher on student-nominated support (offline and online scoring)."""
import torch

PLANNED_SCORER_COST_FACTOR = 1.5

@torch.no_grad()
def content_correction(selected_logp,placeholder_logp,student_prob):
    if selected_logp.shape!=placeholder_logp.shape or selected_logp.shape!=student_prob.shape:
        raise ValueError('The three views must use one candidate support')
    delta=selected_logp.float()-placeholder_logp.float()
    return delta-(student_prob.float()*delta).sum(-1,keepdim=True)
