from game_driver import TrafficWorld
from game_logic import TrafficNetwork
import torch as T
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from nfsp import NFSPAgent
from collections import defaultdict

if __name__ == '__main__':
    train_mode = True
    load_checkpoint = False

    env = TrafficWorld()
    obs_dict = env.reset()

    # Agent instantiation
    #* Player is instantiated inside TrafficWorld, 
    #* external agent makes decisions and passes,
    #* information to the Player in World
    robber = NFSPAgent(start_node=env.robber_player.get_current_node(), player_type=env.robber_player.player_type,
                        input_dims=env.get_obs_shape(env.robber_player.player_type), player_name='robber')
    police1 = NFSPAgent(start_node=env.police_player_1.get_current_node(), player_type=env.police_player_1.player_type,
                        input_dims=env.get_obs_shape(env.police_player_1.player_type), player_name='police1')
    police2 = NFSPAgent(start_node=env.police_player_2.get_current_node(), player_type=env.police_player_2.player_type,
                        input_dims=env.get_obs_shape(env.police_player_2.player_type), player_name='police2')

    player_type_list = [robber, police1]
    agent_list = [robber, police1, police2]

    ######################## END SET-UP CODE ########################

    if train_mode:
        if load_checkpoint:
            for agent in agent_list:
                #! Run count number for testing only
                agent.load_models(1)

        log_dir = '../log_dir/trains' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)
        n_games = 2000
        max_steps = 500
        # police1 and police2 have same score
        best_scores = {'robber':-np.inf, 'police':-np.inf}

        # info storing for logging
        scores, eps_history, steps = defaultdict(list), defaultdict(list), defaultdict(list)
        # agents all have same step count
        n_steps = 0

        # episode loop
        for i in range(n_games):
            print('...beginning training, episode ', i, '...')
            done = False
            obs_dict = env.reset()
            score = {'robber':0, 'police':0}

            # steps loop
            for s in range(max_steps):
                actions = []
                next_nodes = []
                best_responses = []
                for agent in agent_list:
                    obs = obs_dict[agent.player_type]
                    action, next_node, best_response = agent.choose_action(obs)
                    actions.append(action)
                    next_nodes.append(next_node)
                    best_responses.append(best_response)
                obs_dict_, rewards_dict, done = env.step(next_nodes) 

                for player_type in rewards_dict.keys():
                    score[player_type] += rewards_dict[player_type]

                if not load_checkpoint:
                    for idx, agent in enumerate(agent_list):
                        obs = obs_dict[agent.player_type]
                        action = actions[idx]
                        reward = rewards_dict[player_type]
                        obs_ = obs_dict_[agent.player_type]

                        agent.store_rl(obs, action, reward, obs_, int(done))

                        if best_responses[idx]:
                            agent.store_sl(obs, action)
                        
                        agent.learn()

                n_steps += 1

                if done:
                    break

            for agent in player_type_list:
                scores[agent.player_type].append(score[agent.player_type])
                eps_history[agent.player_type].append(agent.epsilon)

            avg_scores = {}

            for agent in player_type_list:
                writer.add_scalar("Score/"+agent.player_type, np.asarray(scores[agent.player_type][i], dtype=np.float32), i)
                writer.add_scalar("Epsilon/"+agent.player_type, np.asarray(eps_history[agent.player_type][i], dtype=np.float32), i)
                avg_scores[agent.player_type] = np.mean(scores[agent.player_type][-100:])
                writer.add_scalar("Avg_Score/"+agent.player_type, np.asarray(avg_scores[agent.player_type], dtype=np.float32), i)
                if avg_scores[agent.player_type] > best_scores[agent.player_type] or i % 100 == 0:
                    #! DEBUG ONLY
                    agent.save_models(1)
                    best_scores[agent.player_type] = avg_scores[agent.player_type]

            print('...episode ', i," completed...")
            print('...steps taken so far ', n_steps, '...')
            print('...agent epsilons ', robber.epsilon, '...')
            print('...robber scores ', scores[robber.player_type][i])
            print('...police scores ', scores[police1.player_type][i])