import ast
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from objective import block


def check():
    torch.manual_seed(42)
    source = Path('/mnt/data/ForkAudit/math-opsd/source/opsd_trainer.py').read_bytes()
    method, = [x for x in ast.walk(ast.parse(source)) if isinstance(x,ast.FunctionDef) and x.name=='generalized_jsd_loss']
    method.decorator_list = []
    namespace = dict(torch=torch,F=F)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'opsd_trainer.py','exec'),namespace)
    official_loss = namespace[method.name]
    result = dict(trainer_sha256=hashlib.sha256(source).hexdigest())
    for dtype in (torch.float64, torch.float32):
        errors = []
        clipped_count = 0
        for width, scale in ((13, 0), (32, 1), (512, 2), (71, 4)):
            s = (torch.randn(97, width, dtype=dtype)*scale).requires_grad_()
            t = torch.randn_like(s)*scale
            lp = (s / 1.1).log_softmax(-1)
            lq = (t / 1.1).log_softmax(-1)
            k = F.kl_div(lp,lq,reduction='none',log_target=True)
            assert (k.detach() != .05).all()
            official, = torch.autograd.grad(official_loss(s,t,beta=0,temperature=1.1,token_clip=.05,reduction='sum'),s)
            value, stats = block(s,t,1.)
            actual, = torch.autograd.grad(value,s)
            assert stats[-1] == 0
            clipped_count += int(stats[0])
            errors.append(float((actual-official).abs().max()))
            # Independent conditional-distribution calculation for both trained arms.
            u = k.detach() <= .05
            p, q = lp.exp(), lq.exp()
            pu, qu = (p*u).sum(-1), (q*u).sum(-1)
            for coefficient in (0.,.5):
                ref = (q*u*(lq-lp+(pu.log()-qu.log())[:,None])).sum()-coefficient*(qu*pu.log()).sum()
                torch.testing.assert_close(block(s,t,coefficient)[0],ref)
        assert max(errors)<(1e-12 if dtype==torch.float64 else 1e-6)
        result[str(dtype)] = dict(max_abs_error=max(errors),rows=388,clipped_elements=clipped_count)
    return result


if __name__=='__main__':
    out=check()
    print(out)
    Path(__file__).with_name('gradient_check.json').write_text(json.dumps(out,indent=2)+'\n')
