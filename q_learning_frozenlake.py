# Reinforcement Learning on FrozenLake using Q-Learning

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create FrozenLake 4x4 environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

# State and action details
state_size = env.observation_space.n      # 16 states
action_size = env.action_space.n          # 4 actions

# Create Q-table
q_table = np.zeros((state_size, action_size))

# Hyperparameters
episodes = 1000
learning_rate = 0.8
discount_rate = 0.95
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
max_steps = 100

# Store rewards
rewards_per_episode = []

# Action symbols
actions = ["←", "↓", "→", "↑"]

# Training using Q-Learning
for episode in range(episodes):
    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):

        # Epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()   # Exploration
        else:
            action = np.argmax(q_table[state])   # Exploitation

        # Perform action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-Learning update formula
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_rate * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    # Reduce exploration slowly
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    rewards_per_episode.append(total_reward)

# Final Q-table
print("\nFinal Q-Table:")
print(q_table)

# Episode comparison
print("\nEpisode Comparison:")
print("Episode 1 Reward:", rewards_per_episode[0])
print("Episode 500 Reward:", rewards_per_episode[499])
print("Episode 1000 Reward:", rewards_per_episode[999])

# Plot reward per episode
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode - FrozenLake Q-Learning")
plt.grid(True)
plt.show()

# Final policy using arrows
print("\nFinal Policy using Arrows:")
for state in range(state_size):
    best_action = np.argmax(q_table[state])
    print(actions[best_action], end=" ")

    if (state + 1) % 4 == 0:
        print()

# Test trained agent
print("\nTesting Trained Agent:")

state, info = env.reset()
total_reward = 0

for step in range(max_steps):
    action = np.argmax(q_table[state])
    next_state, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step + 1}: State {state} | Action {actions[action]} | "
        f"Next State {next_state} | Reward {reward}"
    )

    state = next_state
    total_reward += reward

    if terminated or truncated:
        break

print("\nTotal Reward after Training:", total_reward)

env.close()