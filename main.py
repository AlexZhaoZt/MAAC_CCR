import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
#from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
import time
import matplotlib.pyplot as plt
import pickle

def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    #log_dir = run_dir / 'logs'
    os.makedirs(run_dir)
    #logger = SummaryWriter(str(log_dir))

    # Initialization of evaluation metrics
    collisions = [0]
    success_nums = [0]
    ccr_activates = [0]
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_collisions = []  
    final_ep_activates = []  
    final_ep_success_nums = []

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    
    env = make_env(config.env_id, discrete_action=True)
    num_agents = env.n
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    
    # if config.emergency:
    #     env.switch_emergency()

    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)
    
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0

    #### remove all tensorboard methods, replace with print and pickle
    
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        #print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                ep_i + 1 + config.n_rollout_threads,
        #                                config.n_episodes))
        if config.emergency:
            env.switch_emergency()
        obs = env.reset()
        model.prep_rollouts(device='cpu')
        
        t_start = time.time()
        
        prev_obs = None
        act_n_t_minus_1 = None
        
        for et_i in range(config.episode_length):
            if config.CCR:
                if act_n_t_minus_1:
                    target_obs_n, _, _, _ = env.oracle_step(act_n_t_minus_1)
                    diff_state = obs[:,:,:4] - target_obs_n[:,:,:4] # 12x4x4

                    if config.env_id == 'wall' or config.env_id == 'strong_wind' or config.env_id == 'wall_expos':  
                        diff_obs = obs[:,:,-(model.nagents + 8 + 1)]
                    elif config.env_id == 'turbulence':
                        diff_obs = obs[:,:,-(model.nagents + 2 + 1)]
                    else:
                        assert(False)

                    emerg_n = np.sum(diff_state ** 2, axis=-1) + diff_obs # 12x4
                    
                    env.oracle_update()
                    
                    # obs: 12x4x20
                    # emerg_n: 12x4
                    for agent_i in range(model.nagents):
                        for agent_j in range(model.nagents):
                            #print(obs[:, agent_i, -agent_j])
                            #print(emerg_n[:, agent_j])
                            obs[:, agent_i, -agent_j] = emerg_n[:, agent_j]
                            #print(obs[:, agent_i, -agent_j])
                            #print(emerg_n[:, agent_j])
            # collect experience
            if prev_obs is not None:
                replay_buffer.push(prev_obs, agent_actions, rewards, obs, dones)
            
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables            
            torch_agent_actions = model.step(torch_obs, explore=True)

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            
            next_obs, rewards, dones, infos = env.step(actions)
            
            if config.CCR:
                if act_n_t_minus_1:
                    for i in range(model.nagents):
                        for j in range(model.nagents):
                            # ccr_activates[-1] += 1
                            intrinsic_reward = np.linalg.norm(next_obs[:,i,2:4] - obs[:,j,2:4], axis=-1) - np.linalg.norm(obs[:,i,2:4] - obs[:,j,2:4], axis=-1)
                            intrinsic_reward /= (1 + np.linalg.norm(obs[:,i,2:4] - obs[:,j,2:4], axis=-1))
                            intrinsic_reward *= (emerg_n[:,j] - emerg_n[:,i]) 
                            rewards[:,i] += 10 * intrinsic_reward / np.sqrt(num_agents)
                            """
                            if (len(episode_rewards) == 2 or len(episode_rewards) == 2000 or len(episode_rewards) == 5000) and episode_step % 5 == 0:
                                Ls[i].append('      intrinsic reward = ' + str(intrinsic_reward) + '\n')
                            """
                            # if i == j: continue
                            # emerg_invalid = ~((emerg_n[:,j] > emerg_n[:,i]) & (emerg_n[:,j] > 0))
                            # ccr_activates[-1] += (~emerg_invalid).sum()
                            # intrinsic_reward = np.linalg.norm(next_obs[:,i,2:4] - obs[:,j,2:4], axis=-1) - np.linalg.norm(obs[:,i,2:4] - obs[:,j,2:4], axis=-1)
                            # intrinsic_reward[emerg_invalid] = 0 
                            # rewards[:,i] += 10 * intrinsic_reward
                        
                act_n_t_minus_1 = actions
            
            prev_obs = obs

            obs = next_obs
            
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=None)
                    model.update_policies(sample, logger=None)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
        
        
        
        ls_num_collision = env.get_collision_and_zero_out()
        
        collisions.append(np.array(ls_num_collision).mean())  # might need to convert to np.int

        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        ep_rews = np.array(ep_rews).mean()
        # save model, display training output
        
        print("episodes: {}, mean episode reward: {}, mean number of collisions with wall: {}, ccr activates: {}, success numbers: {}, time: {}".format(
                ep_i, ep_rews, np.mean(collisions[-config.save_rate:]), np.mean(ccr_activates[-config.save_rate:]), np.mean(success_nums[-config.save_rate:]), round(time.time()-t_start, 3)))

        # Keep track of final episode reward
        final_ep_rewards.append(ep_rews)
        # final_ep_activates.append(np.mean(ccr_activates[-config.save_rate:]))
        final_ep_collisions.append(np.mean(collisions[-config.save_rate:]))
        final_ep_success_nums.append(np.mean(success_nums[-config.save_rate:]))
        if ep_i % config.save_rate == 0:
            x_axis = np.arange(0, ep_i + 1, step=12)
            # plot reward data
            rew_file_name = run_dir / 'rewards.png'

            plt.plot(x_axis, final_ep_rewards)
            plt.xlabel('training episode')
            plt.ylabel('reward')
            #plt.legend()
            plt.savefig(rew_file_name)

            plt.clf()

            collision_file_name = run_dir / 'collisions.png'

            plt.plot(x_axis, final_ep_collisions)
            plt.xlabel('training episode')
            plt.ylabel('number of collisions')
            #plt.legend()
            plt.savefig(collision_file_name)

            plt.clf()

            # activates_file_name = run_dir / 'activates.png'

            # plt.plot(x_axis, final_ep_activates)
            # plt.xlabel('training episode')
            # plt.ylabel('CCR activates')
            # #plt.legend()
            # plt.savefig(activates_file_name)

            # plt.clf()

            success_file_name = run_dir / 'successes.png'

            plt.plot(x_axis, final_ep_success_nums)
            plt.xlabel('training episode')
            plt.ylabel('success numbers')
            #plt.legend()
            plt.savefig(success_file_name)

            plt.clf()
            
            rew_file_name = run_dir
            collision_file_name = run_dir
            success_nums_file_name = run_dir
            activates_file_name = run_dir
                
            rew_file_name /= 'rewards.pkl'
            collision_file_name /= 'collisions.pkl'
            success_nums_file_name /= 'success_nums.pkl'
            # activates_file_name /= 'activates.pkl'

            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            with open(collision_file_name, 'wb') as fp:
                pickle.dump(final_ep_collisions, fp)
                
            # with open(activates_file_name, 'wb') as fp:
            #     pickle.dump(final_ep_activates, fp)
                
            with open(success_nums_file_name, 'wb') as fp:
                pickle.dump(final_ep_success_nums, fp)
                
                plt.clf()

      
        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    #logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    #logger.close()
    

def test(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        # runs the newest
        run_num = max(exst_run_nums)

    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    
    # Initialization of evaluation metrics
    collisions = [0]
    success_nums = [0]
    ccr_activates = [0]
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_collisions = []  
    final_ep_activates = []  
    final_ep_success_nums = []

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    
    env = make_env(config.env_id, discrete_action=True)
    env.seed(run_num)
    np.random.seed(run_num)
    model = AttentionSAC.init_from_save(run_dir / 'model.pt', True)
    
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0

    #### remove all tensorboard methods, replace with print and pickle
    
    for ep_i in range(0, config.n_episodes):
        
        obs = np.expand_dims(np.array(env.reset()), 0)
        model.prep_rollouts(device='cpu')
        
        t_start = time.time()
        
        prev_obs = None
        act_n_t_minus_1 = None
        
        for et_i in range(config.episode_length):
            if config.CCR:
                if act_n_t_minus_1:
                    target_obs_n, _, _, _ = env.oracle_step(act_n_t_minus_1[0])
                    
                    target_obs_n = np.expand_dims(np.array(target_obs_n), 0)
                    
                    diff_state = obs[:,:,:4] - target_obs_n[:,:,:4] # 1x4x4

                    if config.env_id == 'wall':  
                        diff_obs = obs[:,:,-(model.nagents + 8 + 1)]
                    elif config.env_id == 'turbulence':
                        diff_obs = obs[:,:,-(model.nagents + 2 + 1)]
                    else:
                        assert(False)

                    emerg_n = np.sum(diff_state ** 2, axis=-1) + diff_obs # 1x4
                    
                    env.oracle_update()
                    
                    # obs: 1x4x20
                    # emerg_n: 1x4
                    for agent_i in range(model.nagents):
                        for agent_j in range(model.nagents):
                            obs[:, agent_i, -agent_j] = emerg_n[:, agent_j]
            
            # collect experience
            if prev_obs is not None:
                replay_buffer.push(prev_obs, agent_actions, rewards, obs, dones)
            
            #print(obs)
            # convert observation to torch Variable
            torch_obs = []
            for i in range(model.nagents):
                torch_obs.append(Variable(torch.Tensor(obs[:, i]),
                                  requires_grad=False))
            # print(torch_obs)
            # get actions as torch Variables            
            torch_agent_actions = model.step(torch_obs, explore=False)

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            
            # rearrange actions to be per environment
            actions = [[ac[0] for ac in agent_actions]]

            # rearrange actions to be per environment
            #actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            
            next_obs, rewards, dones, infos = env.step(actions[0])
            
            next_obs = np.expand_dims(np.array(next_obs),0)
            rewards = np.expand_dims(np.array(rewards),0)
            dones = np.expand_dims(np.array(dones),0)
            infos = np.expand_dims(np.array(infos),0)
            
            if config.CCR:
                act_n_t_minus_1 = actions
            
            prev_obs = obs

            obs = next_obs
            
            t += 1
        
            # for displaying learned policies
            if config.display:
                time.sleep(0.1)
                env.render()
                continue

    env.close()
    #logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    #logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=16000, type=int)
    parser.add_argument("--episode_length", default=35, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--save_rate", default=10, type=int)
    parser.add_argument("--plots-dir", type=str, default="tmp/plot/", help="directory where plot data is saved")
    parser.add_argument("--CCR", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--emergency", action="store_true", default=False)


    config = parser.parse_args()

    if config.display:
        test(config)
    else:
        run(config)
