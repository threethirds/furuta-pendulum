import itertools as it
from pathlib import Path
import time
from typing import Iterator


class BasicOffPolicyCI:
    """Basic Collect & Infer strategy for off policy algorithms"""

    def __init__(self, directory, run_name):
        self.directory = directory
        self.run_name = run_name
        self.folder = Path(directory) / run_name
        self.folder.mkdir(parents=True, exist_ok=True)

    def iter_rollout(self) -> Iterator[int]:

        # no weights for the first rollout
        if not (self.folder / '0.data').exists():
            yield 0

        t_max = max((int(f.stem) for f in self.folder.glob('*.data')), default=0)

        # when i.data and (i+1).policy exist, collect (i+1).data
        for t in it.count(t_max):
            wait_for((self.folder / f'{t + 1}.policy').exists)
            yield t + 1

    def iter_backprop(self) -> Iterator[int]:

        t_max = max((int(f.stem) for f in self.folder.glob('*.policy')), default=0)

        # when i.policy and i.data exists, train (i+1).policy
        for t in it.count(t_max):
            wait_for((self.folder / f'{t}.data').exists)
            yield t + 1

    def data_path(self, t):
        return self.folder / f'{t}.data'

    def policy_path(self, t):
        return self.folder / f'{t}.policy'

    def checkpoint_path(self, t):
        return self.folder / f'{t}.checkpoint'


def wait_for(predicate):
    while not predicate():
        time.sleep(1)
