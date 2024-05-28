import numpy as np

def state_gen(state, action, obs):
    state_out = state[0].tolist()
    state_out.append(action)
    state_out.append(obs)
    state_out = state_out[2:]
    return np.asarray(state_out)

def get_states(batch): 
    states = [i[0] for i in batch]
    state_arr = np.asarray(states)
    state_arr = state_arr.reshape(len(batch), -1)
    return state_arr

def get_actions(batch): 
    actions = [i[1] for i in batch]
    actions_arr = np.asarray(actions)
    actions_arr = actions_arr.reshape(len(batch))
    return actions_arr

def get_rewards(batch): 
    rewards = [i[2] for i in batch]
    rewards_arr = np.asarray(rewards)
    rewards_arr = rewards_arr.reshape(1, len(batch))
    return rewards_arr

def get_next_states(batch): 
    next_states = [i[3] for i in batch]
    next_states_arr = np.asarray(next_states)
    next_states_arr = next_states_arr.reshape(len(batch), -1)
    return next_states_arr
