import gymnasium as gym

from stable_baselines3 import A2C

import numpy as np

from gymnasium.spaces import Discrete
class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return self.disc_to_cont[act]



# env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.make("TB-v1", render_mode="human")
# env = gym.make("Reacher-v4", render_mode="human")
wrapped_env = DiscreteActions(env, [np.array([1,0]), np.array([-1,0]),
                                        np.array([0,1]), np.array([0,-1])])

# model = A2C("MlpPolicy", env, verbose=1)
model = A2C("MlpPolicy", wrapped_env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()