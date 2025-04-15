import os
import json
import numpy as np
import torch as th
import motornet as mn
from tqdm import tqdm
import pickle

from joblib import Parallel, delayed
import multiprocessing

from my_policy import Policy  # the RNN
from my_task import Gribble1999  # the task
from my_loss import cal_loss  # the loss function
from my_utils import (
    save_model,
    print_losses,
    plot_stuff,
    run_episode,
    test,
    plot_training_log,
    plot_simulations,
    plot_activation,
    plot_kinematics,
)  # utility functions

device = th.device("cpu")  # use the cpu not the gpu

effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
env = Gribble1999(effector=effector, max_ep_duration=1.6)
n_units = 100
policy = Policy(env.observation_space.shape[0], n_units, env.n_muscles, device=device)
optimizer = th.optim.Adam(policy.parameters(), lr=10**-3)

condition = 'test1'
batch_size = 3
h = policy.init_hidden(batch_size = batch_size)
obs, info = env.reset(condition          = condition,
                        catch_trial_perc = 0,
                        ff_coefficient   = 0,
                        options={'batch_size': batch_size})

terminated = False

# Initialize a dictionary to store lists
data = {
    'xy': [],
    'obs': [],
    'tg': [],
    'vel': [],
    'all_actions': [],
    'all_hidden': [],
    'all_muscle': [],
    'all_force': [],
}

while not terminated:
    # Append data to respective lists
    data['all_hidden'].append(h[0, :, None, :])
    data['all_muscle'].append(info['states']['muscle'][:, 0, None, :])

    action, h = policy(obs, h)
    obs, _, terminated, _, info = env.step(action=action)

    data['xy'].append(info["states"]["fingertip"][:, None, :])
    data['obs'].append(obs[:, None, :])
    data['tg'].append(info["goal"][:, None, :])
    data['vel'].append(info["states"]["cartesian"][:, None, 2:])  # velocity
    data['all_actions'].append(action[:, None, :])
    data['all_force'].append(info['states']['muscle'][:, 6, None, :])

# Concatenate the lists
for key in data:
    data[key] = th.cat(data[key], axis=1)
for key in data:
    data[key] = th.detach(data[key])

fig, ax = plot_simulations(xy=data['xy'], target_xy=data['tg'], figsize=(8, 6))
fig, ax = plot_activation(data['all_hidden'], data['all_muscle'])
fig, ax = plot_kinematics(all_xy=data["xy"], all_tg=data["tg"], all_vel=data["vel"], all_obs=data["obs"])
