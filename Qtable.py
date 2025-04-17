import numpy as np
import random
from IPython.display import clear_output
import gym
import gym_examples

enviroment = gym.make('gym_examples/GridWorld-v0', size=5000, n=4)
enviroment.reset()

print('Number of states: {}'.format(enviroment.observation_space))
print('Number of actions: {}'.format(enviroment.action_space))


alpha = 0.1
gamma = 0.6
epsilon = 0.1
q_table = np.zeros([enviroment.size, enviroment.size])
print(q_table)

num_of_episodes = 100000

for episode in range(0, num_of_episodes):
    # Reset the enviroment
    state = enviroment.reset()
    print(state)

    # Initialize variables
    reward = 0
    terminated = False

    while not terminated:
        # Take learned path or explore new actions based on the epsilon
        if random.uniform(0, 1) < epsilon:
            action = enviroment.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, terminated, info = enviroment.step(action)

        # Recalculate
        q_value = q_table[state, action]
        max_value = np.max(q_table[next_state])
        new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_value)

        # Update Q-table
        q_table[state, action] = new_q_value
        state = next_state

    if (episode + 1) % 100 == 0:
        clear_output(wait=True)
        print("Episode: {}".format(episode + 1))
        enviroment.render()

print("**********************************")
print("Training is done!\n")
print("**********************************")
total_epochs = 0
total_penalties = 0
num_of_episodes = 100

for _ in range(num_of_episodes):
    state = enviroment.reset()
    epochs = 0
    penalties = 0
    reward = 0

    terminated = False

    while not terminated:
        action = np.argmax(q_table[state])
        state, reward, terminated, info = enviroment.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print("**********************************")
print("Results")
print("**********************************")
print("Epochs per episode: {}".format(total_epochs / num_of_episodes))
print("Penalties per episode: {}".format(total_penalties / num_of_episodes))
