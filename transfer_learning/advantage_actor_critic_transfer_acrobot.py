import gym
import transfer_learning.runners as runners

def discrete_range(range_start=-1.0, range_end=-1.0, bins=10):
    range_size = range_end - range_start
    step_size = range_size/bins
    range_bins = [round(range_start+a*step_size, 1) for a in range(0, bins+1) if a != bins/2]
    return range_bins

# Define all environments
env_cp = gym.make('CartPole-v1')
env_mcc = gym.make('MountainCarContinuous-v0')
env_mcc._max_episode_steps = 2000
env_acrobot = gym.make('Acrobot-v1')

CONTINUOUS_N_OF_BINS = 6
actions_mcc = discrete_range(env_mcc.action_space.low[0],
                             env_mcc.action_space.high[0],
                             CONTINUOUS_N_OF_BINS)

envs_list = [env_cp, env_mcc, env_acrobot]
envs_names_list = ['CP', 'MCC', 'ACROBOT']
state_size_list = [len(env_cp.reset()), len(env_mcc.reset()), len(env_acrobot.reset())]
action_size_list = [env_cp.action_space.n, CONTINUOUS_N_OF_BINS, env_acrobot.action_space.n]
convergence_list = [475, 80, -100]

max_state_size = max(state_size_list)
max_action_size = max(action_size_list)

print('max state size: {}'.format(max_state_size))
print('max action size: {}'.format(max_action_size))

POLICY_DEEP = [True, False]
VALUE_DEEP = [True, False]
# Log path
PATH_LOG_LIST = ['./saved_logs/actor_critic_cp',
                 './saved_logs/actor_critic_mcc',
                 './saved_logs/actor_critic_acrobot']
# Save model path
PATH_SAVE_MODEL_LIST = ['./saved_models/actor_critic_cp',
                        './saved_models/actor_critic_mcc',
                        './saved_models/actor_critic_acrobot']

discount_factor = 0.99
lr_list_policy = [0.001, 0.0001]
lr_list_value = [0.01, 0.005, 0.001]

render = False

runners.actorCriticRunner(env=envs_list[2], env_state_size=state_size_list[2], env_action_size=action_size_list[2],
                          state_size=max_state_size, action_size=max_action_size,
                          POLICY_DEEP=POLICY_DEEP, VALUE_DEEP=VALUE_DEEP, lr_list_policy=lr_list_policy,
                          lr_list_value=lr_list_value, discount_factor=discount_factor, render=render,
                          PATH_LOG=PATH_LOG_LIST[2], PATH_SAVE=PATH_SAVE_MODEL_LIST[2], PATH_LOAD=None,
                          env_name=envs_names_list[2],
                          convergence_amount=convergence_list[2], max_episodes=5000, max_steps=7000, actions_continuous=None)