import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, MultiBinary, Discrete
import machine


class HumanoidEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        #observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        observation_space = MultiBinary(8)
        action_space = Discrete(5,)
        MuJocoPyEnv.__init__(
            self, "/home/kyle/workspaces/TB/Lar_TB/T-bar.xml", 2, observation_space=observation_space, action_space=action_space, **kwargs
        )

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob,
            reward,
            False,
            False,
            {},
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int)