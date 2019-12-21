import tensorflow as tf

from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import logger
from importlib import import_module
import gym
import os

def make_vec_env(env_id, seed):
    """
    Create environment
    """
    env = gym.make(env_id)
    env.seed(seed)
    def make_env(env):
        return lambda: env
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), '0'), allow_early_resets=True)
    set_global_seeds(seed)
    return DummyVecEnv([make_env(env)])


def disc_arg_parser():
    """
    Create an argument parser for DISC
    """
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='Ant-v1')
    parser.add_argument('--leng', help='Replay length', type=int, default=64)
    parser.add_argument('--epsilon', help='Clipping factor', type=float, default=0.4)
    parser.add_argument('--epsilon_b', help='Batch inclusion factor', type=float, default=0.1)
    parser.add_argument('--jtarg', help='IS target constant', type=float, default=0.001)
    parser.add_argument('--gaev', help='use GAE-V', type=int, default=1)
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    parser.add_argument('--alg', help='Algorithm', type=str, default='DISC')
    parser.add_argument('--num_timesteps', type=float, default=3e6),
    parser.add_argument('--log_dir', help='log dir', type=str, default=None)
    args, _ = parser.parse_known_args()

    return parser

def train(args, seed):
    total_timesteps = int(args.num_timesteps)

    learn = import_module('.'.join(['baselines', args.alg, args.alg])).learn

    alg_kwargs={}
    alg_kwargs["lr"] = lambda f: 3e-4 * f
    alg_kwargs["value_network"] = 'copy'
    alg_kwargs["replay_length"] = args.leng
    alg_kwargs["gaev"] = args.gaev
    alg_kwargs["J_targ"] = args.jtarg
    alg_kwargs["epsilon_b"] = args.epsilon_b
    alg_kwargs["epsilon"] = args.epsilon
    alg_kwargs['network'] = 'mlp'
    sess = tf.InteractiveSession()
    env = make_vec_env(args.env, seed)
    eval_env = make_vec_env(args.env, seed)

    print('Training {} on {} with arguments \n{}'.format(args.alg, args.env, alg_kwargs))
    """
    Run DISC training
    """
    model = learn(
        env=env,
        eval_env=eval_env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )
    return model, env, sess, eval_env


def main():
    args, _ = disc_arg_parser().parse_known_args()

    dir = args.log_dir
    if args.log_dir is not None:
        dir = dir + '/seed%d' % args.seed

    logger.configure(dir=dir)

    model, env, sess, evalenv = train(args, args.seed)
    env.close()
    evalenv.close()

if __name__ == '__main__':
    main()
