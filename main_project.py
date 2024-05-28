import numpy as np
import pandas as pd
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import qnetwork
import utils

# Read Trace data
data_in = pd.read_csv(r"DynamicMultiChannelRL-master\dataset\real_data_trace.csv")
data_in = data_in.drop("index", axis=1)

np.random.seed(40)

TIME_SLOTS = 100000
NUM_CHANNELS = 16
memory_size = 1000
batch_size = 32
eps = 0.1
action_size = 16
state_size = 32
learning_rate = 1e-2
gamma = 0.9
hidden_size = 50
pretrain_length = 16
n_episodes = 10

env_model = qnetwork.channel_env(NUM_CHANNELS)
q_network = qnetwork.QNetwork(learning_rate=learning_rate, state_size=state_size, action_size=NUM_CHANNELS, hidden_size=hidden_size, name="ChannelQ_Network")


exp_memory = qnetwork.ExpMemory(in_size=memory_size)

history_input = deque(maxlen=state_size)

for i in range(pretrain_length):
    action = np.random.choice(action_size)
    obs = data_in["channel" + str(action-1)][i]
    history_input.append(action)
    history_input.append(obs)

optimizer = tf.keras.optimizers.Adam(learning_rate)

prob_explore = 0.1
loss_0 = []
avg_loss = []
reward_normalised = []
show_interval = 50


for episode in range(n_episodes):
    total_rewards = 0
    loss_init = 0
    print("-------------Episode "+str(episode)+"-----------")
    for time in range(len(data_in)-pretrain_length):
        prob_sample = np.random.rand()
        state_in = np.array(history_input)
        state_in = state_in.reshape([1,-1])

        if(prob_sample <= prob_explore):
            action = np.random.choice(action_size)
        else:
            action = -1
            q_out = q_network(state_in)
            action = np.argmax(q_out)

        obs = data_in["channel"+str(action)][time+pretrain_length] # Corrected access to columns
        next_state = utils.state_gen(state_in, action, obs)
        reward = obs
        total_rewards += reward
        exp_memory.add((state_in, action, reward, next_state))

        state_in = next_state
        history_input = next_state

        if (time > state_size or episode != 0):
            batch = exp_memory.sample(batch_size)
            states = utils.get_states(batch)
            actions = utils.get_actions(batch)
            rewards = utils.get_rewards(batch)
            next_state = utils.get_next_states(batch)

            actuals_Q = q_network(next_state)
            actuals = rewards + gamma * np.max(actuals_Q, axis=1)
            actuals = actuals.reshape(batch_size)
            loss = q_network.train_step(states, actuals, actions, optimizer)

            loss_init += loss.numpy()

            if(episode == 0):
                loss_0.append(loss)

            if(time % show_interval == 0):
                print("Loss  at (t="+ str(time) + ") = " + str(loss.numpy()))

        if(time == len(data_in)-pretrain_length-1 and episode == 0):
            plt.plot(loss_0)
            plt.xlabel("Iteration")
            plt.ylabel("Q Loss")
            plt.title('Iteration vs Loss (Episode 0)')
            plt.show()

    print("Average Loss: ")
    print(loss_init/(len(data_in)))
    print("Total Reward: ")
    print(total_rewards/len(data_in))
    avg_loss.append(loss_init/(len(data_in)))
    reward_normalised.append(total_rewards/len(data_in))

plt.plot(reward_normalised)
plt.xlabel("Episode")
plt.ylabel("Reward Normalised")
plt.title("Episode vs Reward Normalised")
plt.show()

plt.plot(avg_loss)
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.title("Episode vs Average Loss")
plt.show()
