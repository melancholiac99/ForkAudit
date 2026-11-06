import torch
print(torch.__version__,flush=True)
x=torch.ones((152064,3584),device='cuda',dtype=torch.bfloat16)
torch.cuda.synchronize()
print('passed',flush=True)
