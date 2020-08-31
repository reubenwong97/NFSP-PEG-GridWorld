import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class BaseNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(BaseNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.n_actions = n_actions

         # input channel is batch size
         # change this with changes in input size
        self.conv1 = nn.Conv2d(input_dims[0], 32, 2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 2, stride=1)
        self.conv3 = nn.Conv2d(64, 64, 2, stride=1)
        
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        # last extraction layer before splitting
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        raise NotImplementedError

    def save_checkpoint(self, run_num):
        print('...saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file+'_run_'+str(run_num))

    def load_checkpoint(self, run_num):
        file_name = self.checkpoint_file+'_run_'+str(run_num)
        print(f'...DEBUG: Loading {file_name}')
        self.load_state_dict(T.load(file_name))

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(np.prod(dims.size()))

class DeepQNetwork(BaseNetwork):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, *args, **kwargs):
        super(DeepQNetwork, self).__init__(*args, **kwargs)
        self.rl_loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        conv_state = conv3.view(conv3.size()[0], -1)
        fc1 = F.relu(self.fc1(conv_state))
        actions = F.relu(self.fc2(fc1))

        return actions

class ClassificationNetwork(BaseNetwork): # TODO: reference openspiel nfsp for average policy network
    def __init__(self, *args, **kwargs):
        super(ClassificationNetwork, self).__init__(*args, **kwargs)

        # output layers
        #! previously had error when i sent the base class to device but not the subclasses!!
        #! debugging with print(self.sl_net.device) was misleading since it held attribute from base class!!
        self.to(self.device)
    
    def forward(self, state): # TODO: verify that output should be of shape BS x n_actions
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        conv_state = conv3.view(conv3.size()[0], -1)
        fc1 = F.relu(self.fc1(conv_state))
        action_dist = F.softmax(self.fc2(fc1), dim=0)

        return action_dist

    def sl_loss(self, action_dist, actions):
        probs_with_actions = action_dist.gather(1, actions.unsqueeze(1))
        log_probs = probs_with_actions.log()
        loss = -1 * log_probs.mean()

        return loss