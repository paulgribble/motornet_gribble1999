import os
import json
import numpy as np
import torch as th
import motornet as mn
import pickle

from my_policy import Policy  # the RNN
from my_task import Gribble1999  # the task
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


model_name = 'm0'
batch = 'current'

# run model tests and make plots
data, _ = test(
        "models/" + model_name + "/" + "cfg.json",
        "models/" + model_name + "/" + "weights",
        whichtest = 'test1',
)
plot_stuff(data, "models/" + model_name + "/test1_", batch=batch)

# run model tests and make plots
data, _ = test(
        "models/" + model_name + "/" + "cfg.json",
        "models/" + model_name + "/" + "weights",
        whichtest = 'test2',
)
plot_stuff(data, "models/" + model_name + "/test2_", batch=batch)

# run model tests and make plots
data, _ = test(
        "models/" + model_name + "/" + "cfg.json",
        "models/" + model_name + "/" + "weights",
        whichtest = 'test3',
)
plot_stuff(data, "models/" + model_name + "/test3_", batch=batch)

