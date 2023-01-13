import numpy as np
import random

def get_split_batch(batch):
    """
    memory.sample() returns a batch of experiences, but we want an array
    for each element in the memory (s, a, r, s', done)
    """
    states_mb = np.array([each[0][0] for each in batch])
    # print(states_mb.shape) #shape 64*84*84*1 after reshaping im_final -- 64 is the batch size
    actions_mb = np.array([each[0][1] for each in batch])
    # print(actions_mb.shape)
    rewards_mb = np.array([each[0][2] for each in batch])
    # print(rewards_mb.shape) #shape (64,)
    next_states_mb = np.array([each[0][3] for each in batch])
    # print(next_states_mb.shape)
    dones_mb = np.array([each[0][4] for each in batch])
    return states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb

def OU(action, mu=0, theta=0.15, sigma=0.3):
    noise = theta * (mu - action) + sigma * np.random.randn(1)
    return noise

# def OU(action, mu=0, theta=0.15, sigma=0.3, dt=0.1):
#     noise = theta * (mu - action) * dt + sigma * np.random.randn(1) * np.sqrt(dt)
#     return noise

# class OrnsteinUhlenbeckActionNoise(ActionNoise):
#     def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = x0
#         self.reset()

#     def __call__(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#         return x

#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)