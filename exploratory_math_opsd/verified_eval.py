"""Run the official evaluator and require the adapter to be resident on this vLLM lane."""
import argparse,hashlib,runpy,sys
from pathlib import Path
from vllm import LLM
from launch import save
def main():
    source=Path.cwd()
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint_dir')
    parser.add_argument('--output_file',required=True)
    args,_=parser.parse_known_args()
    original=LLM.generate
    def generate(self,*a,**kw):
        request=kw.get('lora_request')
        if args.checkpoint_dir:
            assert request is not None and Path(request.lora_path).resolve()==Path(args.checkpoint_dir).resolve()
            adapter_digest=hashlib.sha256((Path(args.checkpoint_dir)/'adapter_model.safetensors').read_bytes()).hexdigest()
        else:
            assert request is None
            adapter_digest=None
        output=original(self,*a,**kw)
        loaded=sorted(self.llm_engine.list_loras()) if request is not None else []
        if request is not None:assert request.lora_int_id in loaded
        save(Path(args.output_file+'.adapter.json'),dict(adapter=args.checkpoint_dir,sha256=adapter_digest,resident_ids=loaded))
        return output
    LLM.generate=generate
    sys.path.insert(0,str(source/'eval'))
    runpy.run_path(str(source/'eval/evaluate_math.py'),run_name='__main__')

if __name__=='__main__':main()
