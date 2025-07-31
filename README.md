# Reinforcement-Learning-Assignment
This assignment is part of the AI Explorers course at CodeCraft Works.

## Assignment Steps

1. File name needs to follow previously used conventions.
2. Install gym, pick the FrozenLake-v1 environment, 4x4, non-slippery.
3. Define the function **q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1)**
   1. Initialize a q_table with 0s according to the size of the env's observation space and action space.
   2. Loop for each episode in the total number of episodes.
   3. Follow the pseudocode provided.
   4. Have the function return a final q_table, and rewards.

Here is an example of a Q-learning algorithm in pseudocode form:

For each episode:
  Initialize state S
  For each step in episode:
    Choose action A using ε-greedy strategy
    Take action A, observe reward R and next state S'
    Update Q(S, A) using:
      Q(S, A) ← Q(S, A) + α [R + γ * max(Q(S', a')) - Q(S, A)]
    S ← S'
    If S' is terminal, break
