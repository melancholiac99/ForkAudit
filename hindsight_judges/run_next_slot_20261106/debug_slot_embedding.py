import sys,json,torch
sys.path.insert(0,'/root/autodl-tmp/forkaudit_hindsight/hindsight_judges')
import slot_model_loader as L
s,_=L.load_model('7b','/root/autodl-tmp/models/Qwen2.5-7B-Instruct','/root/autodl-tmp/forkaudit_hindsight')
w=s.model.model.embed_tokens.weight
print('weight',w.shape,w.stride(),w.dtype,w.device,w.data_ptr(),flush=True)
for ids in [[1],[151644],json.load(open('/tmp/slot_first_stem_ids.json'))[:1024]]:
 print('ids',len(ids),min(ids),max(ids),flush=True)
 t=s._ids(ids);torch.cuda.synchronize();print('equal',t.cpu().tolist()==[ids],flush=True)
 z=torch.nn.functional.embedding(t,w);torch.cuda.synchronize();print('output',z.shape,torch.isfinite(z).all().item(),flush=True)
