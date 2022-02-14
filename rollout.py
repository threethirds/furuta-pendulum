from collections import defaultdict
import time

import numpy as np
import torch
from tqdm import trange

from ci.basic_off_policy import BasicOffPolicyCI
from pendulum.mock import MockPendulum
from pendulum.env import FurutaPendulumEnv

run_id = '2021-02-11-run-1'
env = FurutaPendulumEnv(MockPendulum(), steps=1, timestep=120)
ci = BasicOffPolicyCI('./runs', run_id)

for t in ci.iter_rollout():
    print(t)

    # Initial rollout - random policy
    if t == 0:
        rollout_steps = 4500
        π = lambda x: env.action_space.sample()

    # Subsequent rollouts - stored policy
    else:
        rollout_steps = 1500
        π = torch.jit.load(ci.policy_path(t-1))
        π.reset()
        π.eval()

    o = None
    done = True
    rollout = defaultdict(list)
    for step in trange(rollout_steps):

        if done:
            o = env.reset()

            rollout['observation'].append(o),
            rollout['action'].append(np.full(env.action_space.shape, np.nan))
            rollout['reward'].append(np.nan)
            rollout['reset'].append(True)
            rollout['terminal'].append(False)
            rollout['time.monotonic'].append(time.monotonic())

        else:
            a = π(o)
            o, r, done, _ = env.step(a)

            rollout['observation'].append(o)
            rollout['action'].append(a)
            rollout['reward'].append(r)
            rollout['reset'].append(False)
            rollout['terminal'].append(done)
            rollout['time.monotonic'].append(time.monotonic())

    # Store data
    with ci.data_path(t).open('wb') as df:
        np.savez_compressed(df, **{k: np.array(v) for k, v in rollout.items()})

    # Pause for the pendulum to calm down
    time.sleep(3)
