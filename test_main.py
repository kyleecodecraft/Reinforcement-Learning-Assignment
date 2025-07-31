import pytest
import gym
import numpy as np
from main import q_learning

def test_q_table_shape():
    env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")
    q_table, rewards = q_learning(env, num_episodes=100)
    assert q_table.shape == (env.observation_space.n, env.action_space.n)

def test_reward_growth():
    env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")
    _, rewards = q_learning(env, num_episodes=1000)
    avg_last_100 = np.mean(rewards[-100:])
    avg_first_100 = np.mean(rewards[:100])
    assert avg_last_100 >= avg_first_100 * 0.8
