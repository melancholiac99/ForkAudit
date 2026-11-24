"""Check the native optimizer used by the registered ZeRO CPU offload, on a tiny CPU tensor."""
import json
import os
from pathlib import Path
import sys


def main(root):
    os.environ['CUDA_HOME']=str(root/'cuda-12.8')
    os.environ['PATH']=os.pathsep.join((str(root/'conda-env/bin'),str(root/'cuda-12.8/bin'),os.environ['PATH']))
    os.environ['TORCH_EXTENSIONS_DIR']=str(root/'torch-extensions')
    os.environ['TORCH_CUDA_ARCH_LIST']='8.0'
    os.environ.setdefault('MAX_JOBS','2')
    import torch
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    p=torch.nn.Parameter(torch.tensor([1.,-1.]))
    opt=DeepSpeedCPUAdam([p],lr=.01,weight_decay=0.)
    p.grad=torch.tensor([.5,-.5])
    opt.step()
    torch.testing.assert_close(p.detach(),torch.tensor([.99,-.99]),atol=2e-6,rtol=0)
    result=dict(optimizer='DeepSpeedCPUAdam',tensor_device=str(p.device),parameters=p.tolist(),
                policy_updates=0,torch_cuda=torch.version.cuda,gpus_visible=torch.cuda.device_count())
    (root/'run/optimizer_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
