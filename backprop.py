import torch

from ci.basic_off_policy import BasicOffPolicyCI
from pendulum.env import FurutaPendulumEnv
from pendulum.mock import MockPendulum

run_id = '2021-02-11-run-1'
env = FurutaPendulumEnv(MockPendulum(), steps=1, timestep=120)
ci = BasicOffPolicyCI('./runs', run_id)

for t in ci.iter_backprop():

    if t == 0:
        model = model_class.init()
        policy = model.policy
        model.save(ci.checkpoint_path(0))
        torch.jit.save(policy, ci.policy_path(0))
        continue

    buffer = npz_buffer(range(t))
    model = load(ci.checkpoint_path(t - 1))

    # backprop
    new_model = backprop(buffer, model)
    new_policy = new_model.policy

    # store
    new_model.save(ci.checkpoint_path(t))
    torch.jit.save(new_policy, ci.policy_path(t))
