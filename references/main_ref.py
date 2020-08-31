import datetime
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch as T
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from nfsp import NFSPAgent
from collections import defaultdict
from util import load_run_counts, update_json

print("Python version:")
print(sys.version)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

if __name__ == '__main__': 
    ######################## BOILERPLATE SET-UP CODE ########################
    env_name = "../smaller_env/CityGen-Trial2"  # Name of the Unity environment binary to launch
    np.random.seed(1)  # set seed
    train_mode = True  # Whether to run the environment in training or inference mode
    load_checkpoint = False
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(base_port=5006, file_name=env_name,
                side_channels=[engine_configuration_channel])

    env.reset()

    engine_configuration_channel.set_configuration_parameters(width=300, height=300, time_scale=15.0)

    robber_team_name = env.get_agent_groups()[0]
    police_team_name = env.get_agent_groups()[1]
    robber_team_specs = env.get_agent_group_spec(robber_team_name)
    police_team_specs = env.get_agent_group_spec(police_team_name)

    robber_step_result = env.get_step_result(robber_team_name)
    police_step_result = env.get_step_result(police_team_name)

    #* Environment space shapes
    robber_obs = np.concatenate((robber_step_result.obs[0][0], robber_step_result.obs[1][0]))
    robber_obs_shape = robber_obs.shape
    police1_obs = np.concatenate((police_step_result.obs[0][0], police_step_result.obs[1][0]))
    police_obs_shape = police1_obs.shape
    
    #* Agent Intitialisation 
    #! Currently, memory initialised even for inference, to be removed
    robber = NFSPAgent(anticipatory=0.1, rl_mem_size=10000, sl_mem_size=50000, gamma=0.99,
                    epsilon=1.0, rl_lr=0.1, sl_lr=0.005, input_dims=(robber_obs_shape),
                    n_actions=robber_team_specs.action_shape[0], eps_min=0.1, batch_size=128,
                    replace=1000, eps_dec=1e-6, chkpt_dir='../models/', algo='NFSPAgent'+'_Robber',
                    env_name='Interdiction-v0')

    police1 = NFSPAgent(anticipatory=0.1, rl_mem_size=10000, sl_mem_size=50000, gamma=0.99,
                    epsilon=1.0, rl_lr=0.1, sl_lr=0.005, input_dims=(police_obs_shape),
                    n_actions=police_team_specs.action_shape[0], eps_min=0.1, batch_size=128,
                    replace=1000, eps_dec=1e-6, chkpt_dir='../models/', algo='NFSPAgent'+'_Police_1',
                    env_name='Interdiction-v0')

    #* Integers store the index to index out correct agent_id
    agent_list = [(robber, 0, robber_team_name, "robber"), (police1, 0, police_team_name, "police1")]
    run_count_dict = load_run_counts()
    run_count = run_count_dict["train_runs"]

    ######################## END SET-UP CODE ########################

    if train_mode:
        if load_checkpoint:
            for agent, _, _, _ in agent_list:
                agent.load_models(run_count)
        # tensorboard setup
        log_dir = '../log_dir/trains/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)
        n_games = 500
        best_scores = {"robber":-np.inf,
                    "police1":-np.inf}

        # Info storing for logging
        scores, eps_history, steps = defaultdict(list), defaultdict(list), defaultdict(list)
        #* Agents all have the same step count
        n_steps = 0

        # Episode loop
        for i in range(n_games):
            print("...beginning training, episode ", i,"...")
            done = False
            env.reset()
            score = {"robber"  : 0,
                    "police1" : 0}
            start_time = time.time()
            # Steps loop
            while not done:
                #* Gets observations and take action
                # store previous observations so they can be stored in memory later
                prev_obs_list = []
                prev_obs_idx = 0
                for agent, team_idx, team_name, name in agent_list:
                    batched_step_result = env.get_step_result(team_name)
                    id_no = batched_step_result.agent_id[team_idx]
                    raw_obs = batched_step_result.get_agent_step_result(id_no).obs
                    obs = np.concatenate((raw_obs[0], raw_obs[1]), axis=-1)
                    prev_obs_list.append(obs)
                    action, best_response = agent.choose_action(obs)
                    env.set_action_for_agent(team_name, id_no, action)

                env.step()

                #* Gets new observations
                for agent, team_idx, team_name, name in agent_list:
                    obs = prev_obs_list[prev_obs_idx]
                    prev_obs_idx += 1
                    batched_step_result = env.get_step_result(team_name)
                    id_no = batched_step_result.agent_id[team_idx]
                    step_result = batched_step_result.get_agent_step_result(id_no)
                    raw_obs_ = step_result.obs
                    obs_ = np.concatenate((raw_obs_[0], raw_obs_[1]), axis=-1)
                    reward = step_result.reward
                    done = step_result.done
                    score[name] += reward

                    agent.store_rl(obs, action, reward, obs_, int(done))
                    if best_response:
                        agent.store_sl(obs, action)
                    agent.learn()
                    # Track value function for each agent at each step
                    value, _ = agent.q_eval(T.tensor(obs).to(agent.q_eval.device))
                    writer.add_scalar("Value/"+name, value, n_steps)
                    n_steps+=1
                    #* End of steps loop

            for agent, _, _, name in agent_list:
                scores[name].append(score[name])
                eps_history[name].append(agent.epsilon)

            avg_scores = {}
            for agent, _, _, name in agent_list:
                writer.add_scalar("Score/"+name, np.asarray(scores[name][i], dtype=np.float32), i)
                writer.add_scalar("Epsilon/"+name, np.asarray(eps_history[name][i], dtype=np.float32), i)
                avg_scores[name] = np.mean(scores[name][-100:])
                writer.add_scalar("Avg_Score/"+name, np.asarray(avg_scores[name], dtype=np.float32), i)
                if avg_scores[name] > best_scores[name] or i % 100 == 0:
                    agent.save_models(run_count)
                    best_scores[name] = avg_scores[name]

            print('...episode ', i," completed...")
            print('...steps taken so far ', n_steps, '...')
            print('...agent epsilons ', robber.epsilon, '...')
            print('...episode took ', time.time() - start_time, '...')

        run_count += 1
        run_count_dict["train_runs"] = run_count
        update_json(run_count_dict, train_mode=train_mode)

    if not train_mode:
        # Running inference
        for agent, _, _, _ in agent_list:
                agent.load_models(run_count)
        log_dir = '../log_dir/tests/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=log_dir)
        n_games = 85
        best_scores = {"robber":-np.inf,
                    "police1":-np.inf}
        # Steps loop
        while not done:
            #* Gets observations and take action
            # store previous observations so they can be stored in memory later
            start_time = time.time()
            prev_obs_list = []
            prev_obs_idx = 0
            for agent, team_idx, team_name, name in agent_list:
                batched_step_result = env.get_step_result(team_name)
                id_no = batched_step_result.agent_id[team_idx]
                raw_obs = batched_step_result.get_agent_step_result(id_no).obs
                obs = np.concatenate((raw_obs[0], raw_obs[1]), axis=-1)
                prev_obs_list.append(obs)
                action, best_response = agent.choose_action(obs)
                env.set_action_for_agent(team_name, id_no, action)

            env.step()

            #* Gets new observations
            for agent, team_idx, team_name, name in agent_list:
                obs = prev_obs_list[prev_obs_idx]
                prev_obs_idx += 1
                batched_step_result = env.get_step_result(team_name)
                id_no = batched_step_result.agent_id[team_idx]
                step_result = batched_step_result.get_agent_step_result(id_no)
                raw_obs_ = step_result.obs
                obs_ = np.concatenate((raw_obs_[0], raw_obs_[1]), axis=-1)
                reward = step_result.reward
                done = step_result.done
                score[name] += reward

                # Track value function for each agent at each step
                value, _ = agent.q_eval(T.tensor(obs).to(agent.q_eval.device))
                writer.add_scalar("Value/"+name, value, n_steps)
                n_steps+=1
                #* End of steps loop

        for agent, _, _, name in agent_list:
            scores[name].append(score[name])
            eps_history[name].append(agent.epsilon)

        avg_scores = {}
        for agent, _, _, name in agent_list:
            writer.add_scalar("Score/"+name, np.asarray(scores[name][i], dtype=np.float32), i)
            writer.add_scalar("Epsilon/"+name, np.asarray(eps_history[name][i], dtype=np.float32), i)
            avg_scores[name] = np.mean(scores[name][-100:])
            writer.add_scalar("Avg_Score/"+name, np.asarray(avg_scores[name], dtype=np.float32), i)
            if avg_scores[name] > best_scores[name]:
                agent.save_models(run_count)
                best_scores[name] = avg_scores[name]

        print('...episode ', i," completed...")
        print('...steps taken so far ', n_steps, '...')
        print('...agent epsilons ', robber.epsilon, '...')
        print('...episode took ', time.time() - start_time, '...')

    writer.close()
    env.close()