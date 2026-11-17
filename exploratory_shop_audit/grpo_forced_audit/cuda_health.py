"""Small distributed CUDA check; process exit is also checked by the caller."""
import os


def main():
    import torch
    import torch.distributed as dist
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    value=torch.tensor([rank+1.],device='cuda')
    dist.all_reduce(value)
    assert value.item()==sum(range(1,dist.get_world_size()+1))
    x=torch.ones((256,256),device='cuda',dtype=torch.bfloat16)
    assert torch.all(x@x==256)
    torch.cuda.synchronize()
    print(f'CUDA/NCCL rank {rank} passed',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
