import numpy as np
import gym

def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        total_reward = 0

        done = False
        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, _ = env.step(action)

            # Q-value update
            best_next_action = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_action
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return q_table, rewards


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")
    q_table, rewards = q_learning(env)

    print("Final Q-Table:")
    print(q_table)
    print("Average reward over 1000 episodes:", np.mean(rewards))
