"""Auditable process isolation, policy inputs and GPU occupancy accounting."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time

from exploratory_shop_audit.data import stable_seed

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
COMMUNITY = ROOT / '.runtime/external/shop-environment-review/shopping-grpo-longhorizon-13d2b636c0725d4e0c738b8c2eb1eb3b7592bae5'
SOURCE = ROOT / '.runtime/external/shop-environment-review'
CPU_PYTHON = ROOT / '.runtime/shop-review-venv/bin/python'
GPU_PYTHON = ROOT / '.runtime/venvs/verl-agent-20bd331b/bin/python'
MODEL = ROOT / '.runtime/models/Qwen2.5-7B-Instruct'

SYSTEM = '''你是购物助手。根据用户完整要求搜索、浏览商品、选择规格并决定是否购买。你只能使用观察中展示的信息。每次响应可以先简短思考，然后最后单独一行输出 Action: 动作。
动作接口是字符串：search[查询文本]、click[可点击按钮或商品编号或规格值]、finish[no_suitable_product]。观察中 search_products 指 search[...]，其他工具名称只是页面文字，实际使用上述字符串动作。点击参数必须来自当前可点击列表；Buy Now 立即购买当前商品与所选规格并结束。finish 主动结束而不购买。不要同时输出多个动作。最多35次环境动作，重复或无进展操作可能提前结束。'''


def utc():
    return datetime.now(timezone.utc).isoformat()


def append(path, row):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


class RPC:
    def __init__(self, command, log, env=None, timeout=600):
        self.log = Path(log).open('a')
        self.proc = subprocess.Popen([str(x) for x in command], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True,
                                     bufsize=1, env=env, start_new_session=True)
        self.timeout = timeout
        try:
            self.ready = self.read()
            if not self.ready.get('ready'):
                raise RuntimeError(f'RPC did not become ready: {self.ready}')
        except Exception:
            self.close()
            raise

    def read(self):
        # Only one line is emitted per request; no read-ahead buffering ambiguity.
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError(f'RPC timeout pid={self.proc.pid}')
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f'RPC exited {self.proc.poll()}, log={self.log.name}')
        return json.loads(line)

    def call(self, **request):
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
        self.proc.stdin.flush()
        reply = self.read()
        if not reply.get('ok'):
            raise RuntimeError(reply.get('error', str(reply)))
        return reply['result']

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"cmd":"quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=20)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait()
        self.log.close()


def environment(run, name):
    return RPC([CPU_PYTHON, HERE / 'env_worker.py', '--root', COMMUNITY,
                '--products', SOURCE / 'community_products.json', '--index', SOURCE / 'products.sqlite3',
                '--snapshots', run / 'snapshots'], run / f'{name}.log')


def make_input(tokenizer, task, history, observation, limit=8192):
    """History is (observation, exact assistant output); drop whole oldest pairs."""
    def messages(pairs):
        result = [{'role': 'system', 'content': SYSTEM},
                  {'role': 'user', 'content': '完整购物要求：\n' + task}]
        for obs, response in pairs:
            result.extend([{'role': 'user', 'content': obs}, {'role': 'assistant', 'content': response}])
        result.append({'role': 'user', 'content': observation})
        return result

    minimum = messages([])
    minimum_ids = tokenizer.apply_chat_template(minimum, tokenize=True, add_generation_prompt=True)
    if len(minimum_ids) > limit:
        raise ValueError(f'CURRENT_OBSERVATION_OVERFLOW: minimum input {len(minimum_ids)} > {limit}')
    before = tokenizer.apply_chat_template(messages(history), tokenize=True, add_generation_prompt=True)
    kept = list(history)
    removed = []
    ids = before
    while len(ids) > limit:
        removed.append(kept.pop(0))
        ids = tokenizer.apply_chat_template(messages(kept), tokenize=True, add_generation_prompt=True)
    final = messages(kept)
    return {'messages': final, 'text': tokenizer.apply_chat_template(final, tokenize=False, add_generation_prompt=True),
            'token_ids': ids, 'token_count': len(ids), 'original_token_count': len(before),
            'removed_pairs': removed, 'removed_tokens': len(before) - len(ids)}


def parse_generation(text):
    text = text.replace('\\n', '\n').strip()
    if text.startswith('Action: '):
        return text[len('Action: '):].strip()
    if '\nAction: ' in text:
        return text.split('\nAction: ', 1)[1].strip()
    return text


class Model:
    def __init__(self, run):
        self.run = run
        self.start = time.monotonic()
        self.previous = sum(row.get('gpu_hours', 0) for row in read_rows(run / 'gpu_sessions.jsonl'))
        remaining = 16 * 3600 - self.previous * 3600
        if remaining <= 0:
            raise RuntimeError('GPU_BUDGET_EXHAUSTED')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        self.deadline = time.monotonic() + remaining
        self.rpc = None
        self.stopped = threading.Event()
        # Covers model initialization as well as requests; monitor kills our process group only.
        def watchdog():
            while not self.stopped.wait(1):
                if time.monotonic() >= self.deadline:
                    append(run / 'errors.jsonl', {'time': utc(), 'error': 'GPU_BUDGET_EXHAUSTED'})
                    if self.rpc is not None:
                        os.killpg(self.rpc.proc.pid, signal.SIGKILL)
                    return
        self.monitor = threading.Thread(target=watchdog, daemon=True)
        self.monitor.start()
        append(run / 'progress.jsonl', {'time': utc(), 'event': 'gpu_start', 'previous_gpu_hours': self.previous})
        try:
            self.rpc = RPC([GPU_PYTHON, HERE / 'gpu_worker.py', '--model', MODEL], run / 'gpu.log', env)
        except Exception:
            self.close()
            raise

    def hours(self):
        return self.previous + (time.monotonic() - self.start) / 3600

    def generate(self, requests):
        if time.monotonic() >= self.deadline:
            raise RuntimeError('GPU_BUDGET_EXHAUSTED')
        start = time.monotonic()
        for attempt in range(3):
            try:
                output = self.rpc.call(requests=requests)
                append(self.run / 'inference_costs.jsonl', {'time': utc(), 'requests': len(requests),
                       'input_tokens': sum(len(r['token_ids']) for r in requests),
                       'output_tokens': sum(len(r['token_ids']) for r in output),
                       'seconds_including_retries': time.monotonic() - start, 'attempt': attempt,
                       'gpu_hours_cumulative': self.hours()})
                return output
            except Exception as exc:
                append(self.run / 'errors.jsonl', {'time': utc(), 'stage': 'inference', 'attempt': attempt,
                       'error': str(exc), 'requests': requests})
                if attempt == 2 or self.rpc.proc.poll() is not None:
                    raise

    def close(self):
        if self.rpc:
            self.rpc.close()
        if not self.stopped.is_set():
            self.stopped.set()
            append(self.run / 'gpu_sessions.jsonl', {'time': utc(), 'gpu_count': 1,
                   'gpu_hours': (time.monotonic() - self.start) / 3600})
