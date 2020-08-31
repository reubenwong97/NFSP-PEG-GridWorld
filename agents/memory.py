import numpy as np
import itertools

class RLMemory(object):
    def __init__(self, max_size, n_actions, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0 # position of last stored memory
        self.state_memory = np.zeros((self.mem_size, *input_shape), 
                                                dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), 
                                                dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)


    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size) # max amount we can sample to
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class SLMemory(object):
    def __init__(self, max_size, n_actions, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)

    def store_transition(self, state, action):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # Effecient Resevoir Sampling
        # Pseudocode from: http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
        n = self.action_memory.size
        # Initialise reservoir with first batch_size elements of data
        action_res = list(itertools.islice(self.action_memory, 0, batch_size))
        state_res = list(itertools.islice(self.state_memory, 0, batch_size))
        # Until this threshold use traditional sampling
        t = batch_size * 4
        j = batch_size + 1
        idx = batch_size
        while (j < n and j <= t):
            k = np.random.randint(0, j)
            if (k < batch_size):
                action_res[k] = self.action_memory[j]
                state_res[k] = self.state_memory[j]
            j += 1
        # Once gaps become significant, time for gap sampling
        while (j < n):
            # Draw gap size (g) from geometric distribution with probability p = batch_size/j
            p = batch_size / j
            u = np.random.random()
            g = int(np.floor(np.log(u) / np.log(1-p))) # cast to int else indexing error as float
            j = j + g
            if (j < n):
                k = np.random.randint(0, batch_size)
                action_res[k] = self.action_memory[j]
                state_res[k] = self.state_memory[j]
            j += 1
        
        # Return reservoirs
        return state_res, action_res