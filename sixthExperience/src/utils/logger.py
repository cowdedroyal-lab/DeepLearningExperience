import os
import sys
import time
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.start_time = time.time()

    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str = ''):
        for k, v in scalars.items():
            tag = f'{prefix}/{k}' if prefix else k
            self.writer.add_scalar(tag, v, step)

    def close(self):
        self.writer.close()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def log_text(self, tag: str, text: str, step: int = 0):
        self.writer.add_text(tag, text, step)

    def flush(self):
        self.writer.flush()

class ProgressPrinter:
    def __init__(self, total_steps: int):
        self.total = total_steps
        self.last_len = 0

    def update(self, step: int, info: Dict[str, Any]):
        pct = step / max(1, self.total)
        bar_len = 30
        filled = int(pct * bar_len)
        bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
        metrics_str = ' '.join(f'{k}:{v:.4f}' for k, v in info.items())
        line = f'\r{bar} {step}/{self.total} {pct*100:5.1f}% {metrics_str}'
        sys.stdout.write(line)
        sys.stdout.flush()
        self.last_len = len(line)

    def done(self):
        sys.stdout.write('\n')
        sys.stdout.flush()
