import motornet as mn
import torch as th
import numpy as np
from typing import Any
from typing import Union


class Gribble1999(mn.environment.Environment):
    """Gribble 1999 J. Neurophysiol."""

    def __init__(self, *args, **kwargs):
        # pass everything as-is to the parent Environment class
        super().__init__(*args, **kwargs)
        self.__name__ = "Gribble1999"

    def reset(self, *,
              seed: int | None = None,
              max_ep_duration = 1.6,
              ff_coefficient: float = 0.,
              condition: str = 'train',
              catch_trial_perc: float = 50,  # percentage of trials that are no-go catch trials
              is_channel: bool = False,
              K: float = 1,
              B: float = -1,
              tgt_cue_range: Union[list, tuple, np.ndarray] = (0.1, 0.5), # range of times of randomized tgt-cue onset
              go_cue_range:  Union[list, tuple, np.ndarray] = (0.5, 0.9), # range of times of randomized go-cue onset
              options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        self._set_generator(seed)

        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        joint_state: th.Tensor | np.ndarray | None = options.get('joint_state', None)
        deterministic: bool = options.get('deterministic', False)

        self.catch_trial_perc = catch_trial_perc
        self.ff_coefficient = ff_coefficient
        self.tgt_cue_range = tgt_cue_range  # in seconds
        self.go_cue_range = go_cue_range    # in seconds
        self.is_channel = is_channel
        self.K = K
        self.B = B

        if (condition == 'train'):  # train net to reach to random targets

            joint_state = None

            goal = self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0]
            self.goal = goal if self.differentiable else self.detach(goal)

            # specify tgt cue time
            tgt_cue_time = np.random.uniform(self.tgt_cue_range[0], self.tgt_cue_range[1], batch_size)
            self.tgt_cue_time = tgt_cue_time

            # specify go cue time
            go_cue_time = np.random.uniform(self.go_cue_range[0], self.go_cue_range[1], batch_size)
            self.go_cue_time = go_cue_time

        elif (condition == 'test1'):  # 3 elbow-only movements
            # Experiment 1: elbow alone    : (50,60) deg + (20,40,60) deg elbow flexion

            joint_state = th.from_numpy(np.deg2rad(np.array([50, 60])))
            goal_states_joint = th.from_numpy(np.deg2rad(np.array([[50,80],[50,100],[50,120]], dtype=np.float32)))
            batch_size = 3
            goal = self.joint2cartesian(np.concatenate([goal_states_joint, np.zeros_like(goal_states_joint)],axis=1))
            goal = goal[:, :2]

            self.goal = goal if self.differentiable else self.detach(goal)
            joint_state = th.from_numpy(np.tile(joint_state, (batch_size, 1)))

            # same tgt-cue time for all targets
            tgt_cue_time = np.tile(0.30, batch_size)
            self.tgt_cue_time = tgt_cue_time

            # same go-cue time for all targets
            go_cue_time = np.tile(0.60, batch_size)
            self.go_cue_time = go_cue_time

        elif (condition == 'test2'):  # 3 shoulder-only movements
            # Experiment 2: shoulder alone : (10,80) deg + (20,40,60) deg shoulder flexion

            joint_state = th.from_numpy(np.deg2rad(np.array([10, 80])))
            goal_states_joint = th.from_numpy(np.deg2rad(np.array([[30,80],[50,80],[70,80]], dtype=np.float32)))
            batch_size = 3
            goal = self.joint2cartesian(np.concatenate([goal_states_joint, np.zeros_like(goal_states_joint)],axis=1))
            goal = goal[:, :2]

            self.goal = goal if self.differentiable else self.detach(goal)
            joint_state = th.from_numpy(np.tile(joint_state, (batch_size, 1)))

            # same tgt-cue time for all targets
            tgt_cue_time = np.tile(0.30, batch_size)
            self.tgt_cue_time = tgt_cue_time

            # same go-cue time for all targets
            go_cue_time = np.tile(0.60, batch_size)
            self.go_cue_time = go_cue_time

        elif (condition == 'test3'):  # 4 multi-joint movements
            # Experiment 3: elbow,shoulder combinations : (50,70) + (+20,+30), (+20,-30), (-20,+30), (-20,-30)

            joint_state = th.from_numpy(np.deg2rad(np.array([50, 70])))
            goal_states_joint = th.from_numpy(np.deg2rad(np.array([[70,100],[70,40],[30,100],[30,40]], dtype=np.float32)))
            batch_size = 4
            goal = self.joint2cartesian(np.concatenate([goal_states_joint, np.zeros_like(goal_states_joint)],axis=1))
            goal = goal[:, :2]

            self.goal = goal if self.differentiable else self.detach(goal)
            joint_state = th.from_numpy(np.tile(joint_state, (batch_size, 1)))

            # same tgt-cue time for all targets
            tgt_cue_time = np.tile(0.30, batch_size)
            self.tgt_cue_time = tgt_cue_time

            # same go-cue time for all targets
            go_cue_time = np.tile(0.60, batch_size)
            self.go_cue_time = go_cue_time

        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": joint_state})

        self.elapsed = 0.
        action = th.zeros((batch_size, self.muscle.n_muscles)).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        # specify catch trials
        # In the reset, i will check if the trial is a catch trial or not
        catch_trial = np.zeros(batch_size, dtype='float32')
        p = int(np.floor(batch_size * self.catch_trial_perc / 100))
        catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
        self.catch_trial = catch_trial

        # if catch trial, set the go cue time to max_ep_duration
        # thus the network will not see the go-cue
        self.go_cue_time[self.catch_trial == 1] = self.max_ep_duration
        self.go_cue = th.zeros((batch_size, 1)).to(self.device)

        self.tgt_cue = th.zeros((batch_size, 1)).to(self.device)

        obs = self.get_obs(deterministic=deterministic)

        # initial states
        self.init = self.states['fingertip']

        endpoint_load = th.zeros((batch_size, 2)).to(self.device)
        self.endpoint_load = endpoint_load

        info = {
            "states":        self.states,
            "endpoint_load": self.endpoint_load,
            "action":        action,
            "noisy action":  action,  # no noise here so it is the same
            "obs":           obs,
            "goal":          self.goal,
            "tgt_cue_time":  self.tgt_cue_time,
            "go_cue_time":   self.go_cue_time,
        }
        return obs, info

    def step(self, action, deterministic: bool = False, **kwargs):
        self.elapsed += self.dt

        if deterministic is False:
            noisy_action = self.apply_noise(action, noise=self.action_noise)
        else:
            noisy_action = action

        self.effector.step(
            noisy_action, endpoint_load=self.endpoint_load)  # **kwargs

        # Calculate endpoint_load
        vel = self.states["cartesian"][:, 2:]

        self.goal = self.goal.clone()
        self.init = self.init.clone()

        if self.is_channel:  # force channel probe trial

            X2 = self.goal
            X1 = self.init

            # vector that connect initial position to the target
            line_vector = X2 - X1

            xy = self.states["cartesian"][:, 2:]
            xy = xy - X1

            projection = th.sum(line_vector * xy, axis=-1) / th.sum(line_vector * line_vector, axis=-1)
            projection = line_vector * projection[:, None]

            err = xy - projection

            projection = th.sum(line_vector * vel, axis=-1) / th.sum(line_vector * line_vector, axis=-1)
            projection = line_vector * projection[:, None]
            err_d = vel - projection

            F = -1*(self.B*err + self.K*err_d)  # force from the channel wall
            self.endpoint_load = F

        else:
            FF_matvel = th.tensor([[0, 1], [-1, 0]], dtype=th.float32).to("cpu")
            # set endpoint load to zero before go cue
            self.endpoint_load = self.ff_coefficient * (vel@FF_matvel.T)

            mask = self.elapsed < self.go_cue_time
            self.endpoint_load[mask] = 0

        # specify go cue time
        mask = self.elapsed >= (self.go_cue_time + (self.vision_delay-1) * self.dt)
        self.go_cue[mask] = 1

        obs = self.get_obs(action=noisy_action)
        reward = None
        truncated = False
        terminated = bool(self.elapsed >= self.max_ep_duration)
        info = {
            "states":        self.states,
            "endpoint_load": self.endpoint_load,
            "action":        action,
            "noisy action":  noisy_action,
            # update the target depending on the go cue
            "obs":           obs,
            "goal":          self.goal * self.go_cue + self.init * (1-self.go_cue),
            "tgt_cue_time":  self.tgt_cue_time,
            "go_cue_time":   self.go_cue_time,
        }
        return obs, reward, terminated, truncated, info

    def get_proprioception(self):
        mlen = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
        mvel = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
        prop = th.concatenate([mlen, mvel], dim=-1).squeeze(dim=1)
        return self.apply_noise(prop, self.proprioception_noise)

    def get_vision(self):
        vis = self.states["fingertip"]
        return self.apply_noise(vis, self.vision_noise)

    def get_obs(self, action=None, deterministic: bool = False):
        self.update_obs_buffer(action=action)

        # specify tgt cue time
        mask = self.elapsed >= (self.tgt_cue_time + (self.vision_delay-1) * self.dt)
        self.tgt_cue[mask] = 1

        obs_as_list = [
            self.goal * self.tgt_cue + self.states['fingertip'] * (1-self.tgt_cue),
            self.obs_buffer["vision"][0],         # oldest element
            self.obs_buffer["proprioception"][0], # oldest element
            self.go_cue,                          # specify go cue as an input to the network
        ]
        obs = th.cat(obs_as_list, dim=-1)

        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        return obs
