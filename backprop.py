import torch

from ci.basic_off_policy import BasicOffPolicyCI

run_id = '2021-02-11-run-1'
ci = BasicOffPolicyCI('./runs', run_id)
batch_size = 256
gradient_steps = 500

for t in ci.iter_backprop():

    if t == 0:
        model = model_class.init()
        policy = model.policy
        model.save(ci.checkpoint_path(0))
        torch.jit.save(policy, ci.policy_path(0))

    else:

        # Load data and model checkpoint
        buffer = npz_buffer([ci.data_path(i) for i in range(t)])  # frame stack?
        model = load(ci.checkpoint_path(t - 1))

        for _ in range(gradient_steps):

            # sample and backprop
            batch = buffer.sample(batch_size=batch_size)
            new_model = backprop(batch, model)

        # store
        new_policy = new_model.policy
        new_model.save(ci.checkpoint_path(t))
        torch.jit.save(new_policy, ci.policy_path(t))
