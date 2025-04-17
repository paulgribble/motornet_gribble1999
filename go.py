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

def make_test_plots(model_name):

    env = Gribble1999(effector=mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle()))
    l1,l2,dt = env.skeleton.l1, env.skeleton.l2, env.dt
    batch = 'final'

    data, _ = test(
    "models/" + model_name + "/" + "cfg.json",
    "models/" + model_name + "/" + "weights",
    whichtest = 'test1',
    )
    plot_stuff(data, "models/" + model_name + "/test1_", batch=batch, l1=l1, l2=l2, dt=dt)

    # run model tests and make plots
    data, _ = test(
         "models/" + model_name + "/" + "cfg.json",
         "models/" + model_name + "/" + "weights",
         whichtest = 'test2',
    )
    plot_stuff(data, "models/" + model_name + "/test2_", batch=batch, l1=l1, l2=l2, dt=dt)

    # run model tests and make plots
    data, _ = test(
         "models/" + model_name + "/" + "cfg.json",
         "models/" + model_name + "/" + "weights",
         whichtest = 'test3',
    )
    plot_stuff(data, "models/" + model_name + "/test3_", batch=batch, l1=l1, l2=l2, dt=dt)


def train(model_name, n_batch, jobnum):

    device = th.device("cpu")  # use the cpu not the gpu

    # define a two-joint planar arm
    # using a Hill-type muscle model as described in
    # Kistemaker, Wong & Gribble (2010) J. Neurophysiol. 104(6):2985-94
    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())

    # define a task with center-out reaching movements
    # also allows for NF or FF or force-channel probe trials
    # also includes reaching to random targets in the workspace
    # by default 50% no-go catch trials (to help it learn to stay put until the go cue)
    env = Gribble1999(effector=effector, max_ep_duration=1.6)

    # define the RNN
    n_units = 100
    policy = Policy(env.observation_space.shape[0], n_units, env.n_muscles, device=device)

    # define the learning rule for updating RNN weights
    optimizer = th.optim.Adam(policy.parameters(), lr=10**-3)


    # make a directory to store the model info
    if not os.path.exists(f"models/{model_name}"):
        os.mkdir(f"models/{model_name}")


    # TRAIN THE RNN TO REACH TO RANDOM TARGETS

    batch_size = 32  # number of movements in each batch

    # a dictionary to store loss values over training
    losses = {
        "overall": [],
        "position": [],
        "muscle": [],
        "muscle_derivative": [],
        "hidden": [],
        "hidden_derivative": [],
        "jerk": [],
    }

    # train over batches!
    for batch in tqdm(
        iterable      = range(n_batch), 
        desc          = f"model {jobnum:2d}: Training {n_batch} batches of {batch_size}", 
        unit          = "batch", 
        total         = n_batch, 
        position      = jobnum,
        dynamic_ncols = True,
        mininterval   = 5.0,
        leave         = True
    ):
        # forward pass of all movements in the batch
        data = run_episode(
            env,
            policy,
            batch_size,
            catch_trial_perc=50,
            condition="train",  # 'train' means random targets in the arm's workspace
            ff_coefficient=0.0, # NULL FIELD
            detach=False,
        )

        # compute losses
        loss, losses_weighted = cal_loss(data)

        # backward pass & update weights
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)  # important!
        optimizer.step()

        # Update loss values in the dictionary
        losses["overall"].append(loss.item())
        losses["position"].append(losses_weighted["position"].item())
        losses["muscle"].append(losses_weighted["muscle"].item())
        losses["muscle_derivative"].append(losses_weighted["muscle_derivative"].item())
        losses["hidden"].append(losses_weighted["hidden"].item())
        losses["hidden_derivative"].append(losses_weighted["hidden_derivative"].item())
        losses["jerk"].append(losses_weighted["jerk_loss"].item())

    # end of training, save the model and make plots

    # save model
    save_model(env, policy, losses, model_name)
    with open("models/" + model_name + "/" + "data.pkl", "wb") as f:
        pickle.dump(data, f)
    #print_losses(losses_weighted=losses_weighted, model_name=model_name, batch=batch)



if __name__ == "__main__":

    print("All packages imported.")
    print("pytorch version: " + th.__version__)
    print("numpy version: " + np.__version__)
    print("motornet version: " + mn.__version__)

    n_batch  = 20000  # number of batches to train on
    n_models = 10     # train models in parallel
    
    n_cpus = multiprocessing.cpu_count()
    print(f"found {n_cpus} CPUs")
    print(f"training {n_models} models ...")

    if not os.path.exists("models"):
            os.mkdir("models")

    th._dynamo.config.cache_size_limit = 64

    result = Parallel(n_jobs=n_cpus, backend='multiprocessing')(delayed(train)(f"m{iteration}", n_batch, iteration) for iteration in range(n_models))

    print(f"testing & saving plots for {n_models} models ...")

    result = Parallel(n_jobs=n_cpus, backend='multiprocessing')(delayed(make_test_plots)(f"m{iteration}") for iteration in range(n_models))



