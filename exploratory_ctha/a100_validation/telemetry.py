"""W&B and local JSONL/CSV metrics; checkpoint tensors are never uploaded."""
import csv
import json
import netrc
import os
import time
from pathlib import Path


class Metrics:
    def __init__(self, folder, name, config):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.run = None
        authenticated = bool(os.getenv('WANDB_API_KEY'))
        try:
            authenticated = authenticated or bool(netrc.netrc().authenticators('api.wandb.ai'))
        except (OSError, netrc.NetrcParseError):
            pass
        mode = os.getenv('WANDB_MODE') or ('online' if authenticated else 'offline')
        try:
            import wandb
            self.run = wandb.init(project=os.getenv('WANDB_PROJECT', 'ForkAudit'),
                                  entity=os.getenv('WANDB_ENTITY') or None,
                                  group=os.getenv('CTHA_A100_GROUP', 'ctha-a100-engineering-20261113-r1'),
                                  name=name, job_type='engineering_acceptance', config=config,
                                  dir=str(self.folder), mode=mode, save_code=False,
                                  settings=wandb.Settings(init_timeout=30))
            self.run.define_metric('optimizer_step')
            for prefix in ['train/*', 'stability/*', 'parity/*']:
                self.run.define_metric(prefix, step_metric='optimizer_step')
            info = dict(mode=mode, id=self.run.id, url=self.run.url if mode == 'online' else None,
                        directory=self.run.dir, project=self.run.project)
        except Exception as error:
            info = dict(mode='local_only', error=f'{type(error).__name__}: {error}')
        (self.folder / 'tracking.json').write_text(json.dumps(info, indent=2) + '\n')

    def log(self, values):
        row = dict(time=time.time(), **values)
        self.rows.append(row)
        with (self.folder / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if self.run:
            self.run.log(values)

    def table(self, name, rows):
        if not rows:
            return
        columns = sorted(set().union(*(row.keys() for row in rows)))
        with (self.folder / f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        if self.run:
            import wandb
            self.run.log({name: wandb.Table(columns=columns, data=[[r.get(c) for c in columns] for r in rows])})

    def finish(self, success=True):
        if self.rows:
            self.table('metrics', self.rows)
        if self.run:
            self.run.finish(exit_code=0 if success else 1)
