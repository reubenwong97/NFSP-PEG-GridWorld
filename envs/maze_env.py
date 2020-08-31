import numpy as np
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color
from mazelab import BaseEnv
from mazelab import VonNeumannMotion
import gym
from gym.spaces import Box, Discrete
from copy import deepcopy

class SoloMaze(BaseMaze):
    def __init__(self):
        self.x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        super(SoloMaze, self).__init__()

    @property
    def size(self):
        y = deepcopy(self.x)
        return np.expand_dims(y, axis=0).shape

    @property
    def inner_size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, 'free',np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, 'obstacle', np.stack(np.where(self.x == 1), axis=1))
        # no locations specified for agent and goal
        robber = Object('robber', 2, color.agent_robber, False, 'robber',[])
        goal = Object('goal', 4, color.goal, False, 'goal',np.stack(np.where(self.x == 4), axis=1))
        return free, obstacle, robber, goal

class SoloEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        # # Run by run env variables
        self.robber_start_idx = [[3, 3]]

        self.maze = SoloMaze()
        self.motions = VonNeumannMotion()

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        # Players lists for grouping
        self.players = [self.maze.objects.robber]
        self.robber_list = [self.maze.objects.robber]

    def step(self, actions):
        """
        Parameters
        ----------
        actions (Array of VonNeumannMotions) : actions of agents collated together
        Returns
        -------
        obs : observations for agents
        reward: reward for the episode
        done: whether the episode has terminated
        """
        obs = {}
        rewards = {'robber':0, 'police':0}
        done = False
        info = {}

        # Loop checks movement and moves to valid locations
        for idx, action in enumerate(actions):
            motion = self.motions[action]
            player = self.players[idx]
            object_type = player.object_type
            current_position = player.positions[0]
            new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
            valid = self._is_valid(new_position)

            if valid:
                player.positions = [new_position]
            
            elif not valid:
                # -1 reward for going to invalid blocks
                rewards[object_type] -= 0.01
                # print(name + ' hit wall')
                
        if self._is_goal():
            rewards['robber'] += 1
            done = True
            info['log'] = 'robber escaped'
            print('robber escaped')

        # player specific observations
        base_obs = self.maze.to_value()
        obs['robber'] = self._transform(base_obs, 'robber')

        return obs, rewards, done, info

    def reset(self):
        obs = {}
        self.maze.objects.robber.positions = self.robber_start_idx

        base_obs = self.maze.to_value()
        obs['robber'] = self._transform(base_obs, 'robber')

        return obs

    def get_image(self):
        return self.maze.to_rgb()

    def get_spec_image(self, spec_type):
        return self.maze.to_specific_rgb(spec_type)

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.inner_size[0] and position[1] < self.maze.inner_size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]

        return nonnegative and within_edge and passable

    def _is_goal(self):
        goal = False
        robber_pos = self.maze.objects.robber.positions[0]
        for pos in self.maze.objects.goal.positions:
            if robber_pos[0] == pos[0] and robber_pos[1] == pos[1]:
                goal = True
                break

        return goal

    def _transform(self, base_obs, player_type):
        copy = deepcopy(base_obs)
        if player_type == 'robber':
            # hide police locations with free (wrong? otherwise how can robber ever predict police movements)
            copy[copy == 3] = 0
            copy[copy == 5] = 0
        elif player_type == 'police':
            # hide goal from police
            copy[copy == 4] = 0
        
        return np.expand_dims(copy, axis=0)

class HardMaze(BaseMaze):
    def __init__(self):
        self.x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        super(HardMaze, self).__init__()

    @property
    def size(self):
        y = deepcopy(self.x)
        return np.expand_dims(y, axis=0).shape

    @property
    def inner_size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, 'free',np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, 'obstacle', np.stack(np.where(self.x == 1), axis=1))
        # no locations specified for agent and goal
        robber = Object('robber', 2, color.agent_robber, False, 'robber',[])
        police1 = Object('police1', 3, color.agent_police, False, 'police', [])
        police2 = Object('police2', 5, color.agent_police, False, 'police', [])
        goal = Object('goal', 4, color.goal, False, 'goal',np.stack(np.where(self.x == 4), axis=1))
        return free, obstacle, robber, police1, police2, goal

class HardEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        # # Run by run env variables
        self.robber_start_idx = [[3, 3]]
        self.police1_start_idx = [[5, 9]]
        # self.goal_idx = [[6, 3]]
        self.police2_start_idx = [[8, 2]]

        self.maze = HardMaze()
        self.motions = VonNeumannMotion()

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        # Players lists for grouping
        self.players = [self.maze.objects.robber, self.maze.objects.police1, self.maze.objects.police2]
        self.police_list = [self.maze.objects.police1, self.maze.objects.police2]
        self.robber_list = [self.maze.objects.robber]

    def step(self, actions):
        """
        Parameters
        ----------
        actions (Array of VonNeumannMotions) : actions of agents collated together
        Returns
        -------
        obs : observations for agents
        reward: reward for the episode
        done: whether the episode has terminated
        """
        obs = {}
        rewards = {'robber':0, 'police':0}
        done = False
        info = {}

        # Loop checks movement and moves to valid locations
        for idx, action in enumerate(actions):
            motion = self.motions[action]
            player = self.players[idx]
            object_type = player.object_type
            current_position = player.positions[0]
            new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
            valid = self._is_valid(new_position)

            if valid:
                player.positions = [new_position]
            
            elif not valid:
                # -1 reward for going to invalid blocks
                rewards[object_type] -= 0.01
                # print(name + ' hit wall')
                
        if self._is_goal():
            rewards['robber'] += 1
            rewards['police'] -= 1
            done = True
            info['log'] = 'robber escaped'
            print('robber escaped')

        if self._is_caught() and not self._is_goal():
            rewards['police'] += 1
            rewards['robber'] -= 1
            done = True
            info['log'] = 'robber caught'
            print('robber caught')

        # player specific observations
        base_obs = self.maze.to_value()
        obs['robber'] = self._transform(base_obs, 'robber')
        obs['police'] = self._transform(base_obs, 'police')

        return obs, rewards, done, info

    def reset(self):
        obs = {}
        self.maze.objects.robber.positions = self.robber_start_idx
        self.maze.objects.police1.positions = self.police1_start_idx
        self.maze.objects.police2.positions = self.police2_start_idx
        # self.maze.objects.goal.positions = self.goal_idx

        base_obs = self.maze.to_value()
        obs['robber'] = self._transform(base_obs, 'robber')
        obs['police'] = self._transform(base_obs, 'police')

        return obs

    def get_image(self):
        return self.maze.to_rgb()

    def get_spec_image(self, spec_type):
        return self.maze.to_specific_rgb(spec_type)

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.inner_size[0] and position[1] < self.maze.inner_size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]

        return nonnegative and within_edge and passable

    def _is_goal(self):
        goal = False
        robber_pos = self.maze.objects.robber.positions[0]
        for pos in self.maze.objects.goal.positions:
            if robber_pos[0] == pos[0] and robber_pos[1] == pos[1]:
                goal = True
                break

        return goal

    def _is_caught(self):
        caught = False
        robber_pos = self.maze.objects.robber.positions[0]
        for police in self.police_list:
            position = police.positions[0]
            if position[0] == robber_pos[0] and position[1] == robber_pos[1]:
                caught = True
                break

        return caught

    def _transform(self, base_obs, player_type):
        copy = deepcopy(base_obs)
        if player_type == 'robber':
            # hide police locations with free (wrong? otherwise how can robber ever predict police movements)
            copy[copy == 3] = 0
            copy[copy == 5] = 0
        elif player_type == 'police':
            # hide goal from police
            copy[copy == 4] = 0
        
        return np.expand_dims(copy, axis=0)

class Maze(BaseMaze):
    def __init__(self):
        # self.x = np.array([[1, 1, 1, 1, 1],
        #                 [1, 0, 0, 0, 1],
        #                 [1, 0, 0, 0, 1],
        #                 [1, 0, 0, 0, 1],
        #                 [1, 1, 1, 1, 1]])\
        self.x = np.array([[1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1]
        ])
        super(Maze, self).__init__()
        print(self.objects)

    @property
    def size(self):
        y = deepcopy(self.x)
        return np.expand_dims(y, axis=0).shape

    @property
    def inner_size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, 'free', np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, 'obstacle', True, np.stack(np.where(self.x == 1), axis=1))
        # no locations specified for agent and goal
        robber = Object('robber', 2, color.agent_robber, False, 'robber', [])
        police = Object('police', 3, color.agent_police, False, 'police',[])
        goal = Object('goal', 4, color.goal, False, 'goal',[])
        return free, obstacle, robber, police, goal

class Env(BaseEnv):
    def __init__(self):
        super().__init__()

        # Run by run env variables
        self.robber_start_idx = [[2, 2]]
        self.police_start_idx = [[5, 4]]
        self.goal_idx = [[6, 3]]

        self.maze = Maze()
        self.motions = VonNeumannMotion()

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        # Players lists for grouping
        self.players = [self.maze.objects.robber, self.maze.objects.police]
        self.police_list = [self.maze.objects.police]
        self.robber_list = [self.maze.objects.robber]

    def step(self, actions):
        """
        Parameters
        ----------
        actions (Array of VonNeumannMotions) : actions of agents collated together
        Returns
        -------
        obs : observations for agents
        reward: reward for the episode
        done: whether the episode has terminated
        """
        obs = {}
        rewards = {'robber':0, 'police':0}
        done = False
        info = {}

        # Loop checks movement and moves to valid locations
        for idx, action in enumerate(actions):
            motion = self.motions[action]
            player = self.players[idx]
            name = player.name
            current_position = player.positions[0]
            new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
            valid = self._is_valid(new_position)

            if valid:
                player.positions = [new_position]
            
            elif not valid:
                # -1 reward for going to invalid blocks
                rewards[name] -= 0.01
                # print(name + ' hit wall')
                
        if self._is_goal():
            rewards['robber'] += 1
            rewards['police'] -= 1
            done = True
            info['log'] = 'robber escaped'
            print('robber escaped')

        if self._is_caught() and not self._is_goal():
            rewards['police'] += 1
            rewards['robber'] -= 1
            done = True
            info['log'] = 'robber caught'
            print('robber caught')

        # player specific observations
        base_obs = self.maze.to_value()
        obs['robber'] = self._transform(base_obs, 'robber')
        obs['police'] = self._transform(base_obs, 'police')

        return obs, rewards, done, info

    def reset(self):
        print('reset called')
        obs = {}
        self.maze.objects.robber.positions = self.robber_start_idx
        self.maze.objects.police.positions = self.police_start_idx
        self.maze.objects.goal.positions = self.goal_idx

        base_obs = self.maze.to_value()
        obs['robber'] = self._transform(base_obs, 'robber')
        obs['police'] = self._transform(base_obs, 'police')

        return obs

    def get_image(self):
        return self.maze.to_rgb()

    def get_spec_image(self, spec_type):
        return self.maze.to_specific_rgb(spec_type)

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.inner_size[0] and position[1] < self.maze.inner_size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]

        return nonnegative and within_edge and passable

    def _is_goal(self):
        goal = False
        robber_pos = self.maze.objects.robber.positions[0]
        for pos in self.maze.objects.goal.positions:
            if robber_pos[0] == pos[0] and robber_pos[1] == pos[1]:
                goal = True
                break

        return goal

    def _is_caught(self):
        caught = False
        robber_pos = self.maze.objects.robber.positions[0]
        for police in self.police_list:
            position = police.positions[0]
            if position[0] == robber_pos[0] and position[1] == robber_pos[1]:
                caught = True

        return caught

    def _transform(self, base_obs, player_type):
        copy = deepcopy(base_obs)
        if player_type == 'robber':
            # hide police locations with free (wrong? otherwise how can robber ever predict police movements)
            copy[copy == 3] = 0
        elif player_type == 'police':
            # hide goal from police
            copy[copy == 4] = 0
        
        return np.expand_dims(copy, axis=0)