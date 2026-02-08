import argparse
from distutils.util import strtobool
import os
from pathlib import Path
import time

import random
import numpy as np
import torch
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=Path(__file__).stem,
    # parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).removesuffix(".py"),
                        # __file__ gives path to script file being run
                        # os.path.basename() returns 'tail of path string
                        # removesuffix() removes the '.py' suffix
                        help='experiment name')
    parser.add_argument('--gym-id', type=str, default="CartPole-v1",
                        help='id of gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='optimizer learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='experiment seed')
    parser.add_argument('--total_timesteps', type=int, default=25000,
                        help='total time steps of the experiment')
    parser.add_argument('--torch-deterministic', type=str_to_bool, default=True, nargs='?', const=True,
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
    parser.add_argument('--wandb-project-name', type=str, default='cleanrl',
                        help='Weights and Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Weights and Biases entity (team), defaults to logged-in username')
    args = parser.parse_args()
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