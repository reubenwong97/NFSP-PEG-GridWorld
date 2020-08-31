import numpy as np
import sys
import gym
from envs.maze_env import Env, HardEnv, SoloEnv
import matplotlib.pyplot as plt
import torch as T
import datetime
from torch.utils.tensorboard import SummaryWriter
from agents.nfsp import NFSPAgent
from collections import defaultdict
from utils.runs import Runs
import os
from utils.plotting import plot_scores

if __name__ == '__main__':
    # register gym env
    test_mode = False

    train_mode = True
    load_checkpoint = False

    env_id = 'SoloMaze13x13-v0'
    gym.envs.register(id=env_id, entry_point=SoloEnv, max_episode_steps=250)
    
    env = gym.make(env_id)

    if test_mode:
        models = []
        for i in range(2):
            robber = NFSPAgent(player_type='robber', player_name='robber1', input_dims=(env.observation_space.shape),
                        n_actions=env.action_space.n, env_name=env_id, train=False)
            robber.load_models(0)
            if i == 0:
                models.append(robber.q_eval)
            elif i == 1:
                models.append(robber.q_next)

        model1 = models[0]
        model2 = models[1]

        def check_models(model1, model2):
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                if p1.data.ne(p2.data).sum() > 0:
                    return False

            return True

        print("Models same? ", check_models(model1, model2))

    if train_mode:
        robber = NFSPAgent(player_type='robber', player_name='robber1', input_dims=(env.observation_space.shape),
                        n_actions=env.action_space.n, env_name=env_id, train=True)
        # police1 = NFSPAgent(player_type='police', player_name='police1', input_dims=(env.observation_space.shape),
        #                 n_actions=env.action_space.n, env_name=env_id, train=True)
        # police2 = NFSPAgent(player_type='police', player_name='police2', input_dims=(env.observation_space.shape),
        #                 n_actions=env.action_space.n, env_name=env_id, train=True)

        # for convenience
        players_scoring_list = [robber]
        players_list = [robber]
        # players_scoring_list = [robber, police1]
        # players_list = [robber, police1, police2]
        # for tracking purposes
        run = Runs()
        run.update_runs(111111)

        if load_checkpoint:
            for player in players_list:
                player.load_models(9)

        log_dir = '/log_dir/trains/trains' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)
        n_games = 250
        max_steps = 250
        # best_scores = {'robber':-np.inf, 'police':-np.inf}
        best_scores = {'robber':-np.inf}

        scores, eps_history, steps = defaultdict(list), defaultdict(list), defaultdict(list)
        n_steps = 0

        for i in range(n_games):
            print('...beginning training, episode ', i, '...')
            done = False
            obs_dict = env.reset()
            # img = env.render(mode='human')
            score = {'robber':0, 'police':0}

            for s in range(max_steps):
                img = env.render(mode='human')
                actions = []
                best_responses = []
                for player in players_list:
                    obs = obs_dict[player.player_type]
                    action, best_response = player.choose_action(obs)
                    actions.append(action)
                    best_responses.append(best_response)
                obs_dict_, rewards_dict, done, _ = env.step(actions)

                for player_type in rewards_dict.keys():
                    score[player_type] += rewards_dict[player_type]

                for idx, player in enumerate(players_list):
                    obs = obs_dict[player.player_type]
                    action = actions[idx]
                    reward = rewards_dict[player.player_type]
                    obs_ = obs_dict_[player.player_type]

                    player.store_rl(obs, action, reward, obs_, int(done))

                    if best_responses[idx]:
                        player.store_sl(obs, action)

                    player.learn()

                obs_dict = obs_dict_
                n_steps += 1

                if done:
                    break

            for player in players_scoring_list:
                scores[player.player_type].append(score[player.player_type])
                eps_history[player.player_type].append(score[player.player_type])

            avg_scores = {}

            for player in players_scoring_list:
                writer.add_scalar("Score/"+player.player_type, np.asarray(scores[player.player_type][i], dtype=np.float32), i)
                # writer.add_scalar("Epsilon/"+player.player_type, np.asarray(eps_history[player.player_type][i], dtype=np.float32), i)
                avg_scores[player.player_type] = np.mean(scores[player.player_type][-100:])
                writer.add_scalar("Avg_Score/"+player.player_type, np.asarray(avg_scores[player.player_type], dtype=np.float32), i)
                if avg_scores[player.player_type] > best_scores[player.player_type] or i % 100 == 0:
                    player.save_models(run.count)
                    best_scores[player.player_type] = avg_scores[player.player_type]

            print('...episode ', i," completed...")
            print('...steps taken so far ', n_steps, '...')
            print('...agent epsilons ', robber.epsilon, '...')
            print('...robber scores ', scores[robber.player_type][i])
            # print('...police scores ', scores[police1.player_type][i])

        plot_scores(scores['robber'], n_games)
        # plot_scores(scores['police'], n_games)

        # After the entire training loop, update our run counter
        run.update_runs()

        writer.close()

    elif not train_mode:        
        robber = NFSPAgent(player_type='robber', player_name='robber1', input_dims=(env.observation_space.shape),
                        n_actions=env.action_space.n, env_name=env_id, train=False)
        police1 = NFSPAgent(player_type='police', player_name='police1', input_dims=(env.observation_space.shape),
                        n_actions=env.action_space.n, env_name=env_id, train=False)
        police2 = NFSPAgent(player_type='police', player_name='police2', input_dims=(env.observation_space.shape),
                        n_actions=env.action_space.n, env_name=env_id, train=False)

        # for convenience
        players_list = [robber, police1, police2]

        frames = []

        n_games = 15

        try:
            from PIL import Image
        except:
            raise ImportError

        #* Loading model from run 1
        for player in players_list:
            player.load_models(9)

        log_dir = 'log_dir/evals/eval' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)

        score = {'robber':0, 'police':0}
        scores = []

        done = False

        for i in range(n_games):
            obs_dict = env.reset()

            while not done:
                frames.append(Image.fromarray(env.render(mode='rgb_array')))
                actions = []
                best_responses = []
                for player in players_list:
                    obs = obs_dict[player.player_type]
                    action, _ = player.choose_action(obs)
                    actions.append(action)
                obs_dict_, rewards_dict, done, info = env.step(actions)

                for player_type in rewards_dict.keys():
                    score[player_type] += rewards_dict[player_type]

                obs_dict = obs_dict_

                if done:
                    break
                
            scores.append(score)
            print(score)
            print(info)

            #save gif
            with open('recording/eval'+str(i)+'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.gif', 'wb') as f:
                im = Image.new('RGB', frames[0].size)
                im.save(f, save_all=True, append_images=frames)

        # clean scores
        robber_scores = []
        police_scores = []
        for score in scores:
            robber_scores.append(score['robber'])
            police_scores.append(score['police'])

        # plot_scores(robber_scores, n_games)
        # plot_scores(police_scores, n_games)

        writer.close()

    else:
        raise Exception("train_mode is a boolean!")