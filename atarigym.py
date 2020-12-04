import gym
import numpy as np
# limitar velocidad 1/2
import time

env = gym.make('Pong-v0')
env.reset()

for _ in range(1000):
    env.render()

    # Limitar velocidad 2/2
    time.sleep(0.05)

    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action

env.close()