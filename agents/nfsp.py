import numpy as np
import torch as T
from .networks import DeepQNetwork, ClassificationNetwork
from .memory import RLMemory, SLMemory

class Player(object):
    # Player class separation because Player is all the game engine needs, since it doesnt do learning => wrong to instantiate NFSPAgent in game_driver
    def __init__(self, player_type, player_name):
        self.player_type = player_type
        self.player_name = player_name


class Agent(Player):
    # Agent parent class
    def __init__(self, input_dims, n_actions, chkpt_dir='models', rl_lr=0.1, sl_lr=0.005, epsilon=1.0, eps_min=0.1, eps_dec=1e-4,
                    gamma=0.99, batch_size=32, replace=1000, algo='DqnNFSP', env_name='test', *args, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)
        #* RL Variables
        self.rl_lr = rl_lr
        self.sl_lr = sl_lr
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.input_dims = input_dims
        self.chkpt_dir = chkpt_dir
        self.gamma = gamma
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.n_actions = n_actions
        # Action space is [0, 1, 2]
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0 # number of times we call learn, and replace target Q params

    #* RL functions
     # different agents had different implementation
    def choose_action(self, observation):
        raise NotImplementedError

    def replace_target_network(self):
        raise NotImplementedError

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        raise NotImplementedError

    def load_models(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

class NFSPAgent(Agent):
    def __init__(self, train=True, anticipatory=0.1, rl_mem_size=100000, sl_mem_size=1000000, *args, **kwargs):
        super(NFSPAgent, self).__init__(*args, **kwargs)
        self.anticipatory = anticipatory
        self.train = train

        if self.train:
            self.q_eval = DeepQNetwork(lr=self.rl_lr, n_actions=self.n_actions, 
                                        name=self.env_name+'_'+self.algo+self.player_name+'_q_eval', 
                                        input_dims=self.input_dims, chkpt_dir=self.chkpt_dir).train()
            self.q_next = DeepQNetwork(lr=self.rl_lr, n_actions=self.n_actions, 
                                        name=self.env_name+'_'+self.algo+self.player_name+'_q_next', 
                                        input_dims=self.input_dims, chkpt_dir=self.chkpt_dir).train()

            self.sl_net = ClassificationNetwork(lr=self.sl_lr, n_actions=self.n_actions,
                                name=self.env_name+'_'+self.algo+self.player_name+'_sl_net',
                                input_dims=self.input_dims, chkpt_dir=self.chkpt_dir).train()

        else:
            self.q_eval = DeepQNetwork(lr=self.rl_lr, n_actions=self.n_actions, 
                                name=self.env_name+'_'+self.algo+self.player_name+'_q_eval', 
                                input_dims=self.input_dims, chkpt_dir=self.chkpt_dir).eval()
            self.q_next = DeepQNetwork(lr=self.rl_lr, n_actions=self.n_actions, 
                                        name=self.env_name+'_'+self.algo+self.player_name+'_q_next', 
                                        input_dims=self.input_dims, chkpt_dir=self.chkpt_dir).eval()

            self.sl_net = ClassificationNetwork(lr=self.sl_lr, n_actions=self.n_actions,
                                name=self.env_name+'_'+self.algo+self.player_name+'_sl_net',
                                input_dims=self.input_dims, chkpt_dir=self.chkpt_dir).eval()

        # Initialise buffers
        self.rl_mem = RLMemory(rl_mem_size, self.n_actions, self.input_dims)
        self.sl_mem = SLMemory(sl_mem_size, self.n_actions, self.input_dims)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def choose_action(self, observation):
        # RL Acting
        if np.random.random() > self.anticipatory:
            # CNN expects input shape of BS x input_dims
            if np.random.random() < self.epsilon:
                action = np.random.choice(self.action_space)
            else:
                with T.no_grad():
                    observation = T.tensor(observation, dtype=T.float).to(self.q_eval.device)
                    observation = observation.unsqueeze(0) 
                    actions = self.q_eval.forward(observation)
                    action = T.argmax(actions).item()

            best_response = True

        else:
        # SL Acting
            with T.no_grad():
                observation = T.tensor(observation, dtype=T.float).to(self.sl_net.device)
                observation = observation.unsqueeze(0) 
                distribution = self.sl_net.forward(observation)
                action = distribution.multinomial(1).item() 

                best_response = False

        return action, best_response

    def sample_rl_mem(self, batch_size):
        # No to the below, we want to return pytorch tensors
        # return self.memory.sample_buffer(batch_size)
        state, action, reward, state_, done = self.rl_mem.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def sample_sl_mem(self, batch_size):
        state, action = self.sl_mem.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.sl_net.device) # Send to device, only have 1 gpu anyway
        actions = T.tensor(action).to(self.sl_net.device)

        return states, actions

    def save_models(self, run_num):
        self.q_eval.save_checkpoint(run_num)
        self.q_next.save_checkpoint(run_num)
        self.sl_net.save_checkpoint(run_num)

    def load_models(self, run_num):
        self.q_eval.load_checkpoint(run_num)
        self.q_next.load_checkpoint(run_num)
        self.sl_net.load_checkpoint(run_num)

    def store_rl(self, obs, action, reward, obs_, done):
        self.rl_mem.store_transition(obs, action, reward, obs_, done)

    def store_sl(self, obs, action):
        self.sl_mem.store_transition(obs, action)

    def learn(self):
        if self.rl_mem.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        self.sl_net.optimizer.zero_grad()

        # RL Network Learning
        self._update_rl_network()

        # SL Network Learning
        self._update_sl_network()

    def _update_rl_network(self):
        states, actions, rewards, states_, dones = self.sample_rl_mem(self.batch_size)
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        rl_loss = self.q_eval.rl_loss(q_target, q_pred).to(self.q_eval.device)
        rl_loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def _update_sl_network(self):
        states, actions = self.sample_sl_mem(self.batch_size)
        action_dist = self.sl_net.forward(states)

        sl_loss = self.sl_net.sl_loss(action_dist, actions).to(self.sl_net.device)

        sl_loss.backward()
        self.sl_net.optimizer.step()
