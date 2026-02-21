import argparse
from distutils.util import strtobool
import os
from pathlib import Path
import random
import time

import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)  # initialize environment
        env = gym.wrappers.RecordEpisodeStatistics(env) # records episode statistics (e.g. episodic return) in 'info'
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, "videos/{run_name}", step_trigger=lambda t: t % 1000 == 0) # record video of agent playing game (record video every 1000 timesteps)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


""" PPO Implementation Detail #2: orthogonal initialization of weights and constant initialization of biases """
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0), 
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),  # smaller std to ensure layer parameters have similar scalar values such that probability of taking each action will be similar
        )
    
    def get_value(self, x): 
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)  # unnormalized action probabilities
        probs = Categorical(logits=logits)  # essentially softmax to get normalized action probabilities (action probability distribution)
        if action is None:  # i.e. we are in rollout phase 
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=Path(__file__).stem,
    # parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).removesuffix(".py"),
                        # __file__ gives path to script file being run
                        # os.path.basename() returns 'tail of path string
                        # removesuffix() removes the '.py' suffix
                        help='experiment name')
    parser.add_argument('--gym_id', type=str, default="CartPole-v1",
                        help='id of gym environment')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                        help='optimizer learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='experiment seed')
    parser.add_argument('--total_timesteps', type=int, default=25000,
                        help='total time steps of the experiment')
    parser.add_argument('--torch_deterministic', type=str_to_bool, default=True, nargs='?', const=True,
                        # `type` accepts any callable object i.e. any function that takes a single string 
                        # and returns a value. Ex. type=str returns str('value') and type=int returns int('value')
    # parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        # strtobool(x) returns integer 1 if x is 'y', 'yes', 't', 'true', 'on', or '1'
                        #               and integer 0 if x is 'n', 'no', 'f', 'false', 'off', or '0'
                        # bool(integer 1 or 0) = True or False
                        # default=True means variable will be True by default
                        # nargs='?' tells parser that flag can have 0 or 1 value following it
                        # const=True is the 'constant' value used if flag is provided but no value follows 
                        #               (e.g. just typing --torch-deterministic)
                        help='if toggled or set to True, use deterministic algorithms in Pytorch (defaults to True)')
    parser.add_argument('--cuda', type=str_to_bool, default=True, nargs='?', const=True,
    # parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled or set to True, use CUDA (defaults to True)')
    parser.add_argument('--track', type=str_to_bool, default=False, nargs='?', const=True,
                        help='if toggled or set to True, this experiment will be tracked with Weights and Biases (defaults to False)')
    parser.add_argument('--wandb_project_name', type=str, default='cleanrl',
                        help='Weights and Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights and Biases entity (team), defaults to logged-in username')
    parser.add_argument('--capture_video', type=str_to_bool, default=False, nargs='?', const=True,
                        help='if toggled or set to True, saves videos of agent performance to `videos` directory (defaults to False)')
    
    # Algorithm-specific arguments
    parser.add_argument('--num_envs', type=int, default=4,
                        help='number of parallel environments in SyncVectorEnv')
    parser.add_argument('--num_steps', type=int, default=128,
                        help='number of steps to run in each environment per policy rollout')
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)  # batch size = num_envs * num_steps = 4 * 128 = 512

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    run_name = f"{args.gym_id}__{args.exp_name}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")  #TODO: add "mps" option

    """ PPO Implementation Detail #1: Vectorized Environment Architecture """
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    ) # use SyncVectorEnv API to create vector environment from list of env-creating functions
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space supported currently"
    print("envs.single_observation_space.shape: ", envs.single_observation_space.shape)     # number of features in observation space
    print("envs.single_action_space.n: ", envs.single_action_space.n)                       # number of discrete actions available

    agent = Agent(envs).to(device)
    print(agent)

    """ PPO Implementation Details #3: Adam Optimizer Epsilon Parameter 1e-5 """
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage variable setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # results in tensor of shape (args.num_steps, args.num_envs, [unpacked dimensions of single observation space])
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0                                         # track number of environment steps
    start_time = time.time()                                # for calculating frames per second
    next_obs = torch.Tensor(envs.reset()).to(device)        # store initial observation
    next_done = torch.zeros(args.num_envs).to(device)       # store initial termination status to be false
    num_updates = args.total_timesteps // args.batch_size   # number of iterations/updates required to complete specified training run ... 25000/512=48
    print(num_updates)
    print("next_obs.shape: ", next_obs.shape)                                   # [num_envs, (unpacked observation_space_shape)]
    print("agent.get_value(next_obs): ", agent.get_value(next_obs))             # 1 scalar value for each environment
    print("agent.get_value(next_obs.shape: ", agent.get_value(next_obs).shape)  # (num_envs, 1)
    print()
    print("agent.get_action_and_value(next_obs)", agent.get_action_and_value(next_obs))