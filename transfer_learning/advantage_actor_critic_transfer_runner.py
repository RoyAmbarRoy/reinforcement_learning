import gym
# import transfer_learning.agents.finetune_agent as runners
from transfer_learning.agents import finetune_agent, progressive_networks_agent
import tensorflow as tf


def discrete_range(range_start=-1.0, range_end=-1.0, bins=10):
    range_size = range_end - range_start
    step_size = range_size/bins
    range_bins = [round(range_start+a*step_size, 1) for a in range(0, bins+1) if a != bins/2]
    return range_bins

# Initialize all environments
env_cp = gym.make('CartPole-v1')
env_mcc = gym.make('MountainCarContinuous-v0')
env_mcc._max_episode_steps = 2000
env_acrobot = gym.make('Acrobot-v1')

CONTINUOUS_N_OF_BINS = 6
actions_mcc = discrete_range(env_mcc.action_space.low[0],
                             env_mcc.action_space.high[0],
                             CONTINUOUS_N_OF_BINS)

envs_list = [env_cp, env_mcc, env_acrobot]
state_size_list = [len(env_cp.reset()), len(env_mcc.reset()), len(env_acrobot.reset())]
action_size_list = [env_cp.action_space.n, CONTINUOUS_N_OF_BINS, env_acrobot.action_space.n]
max_state_size = max(state_size_list)
max_action_size = max(action_size_list)

envs_convergence_list = [475, 90, -90]
max_episodes_list = [5000, 3000, 100]
max_steps_list = [7000, 7000, 7000]

print('max state size: {}'.format(max_state_size))
print('max action size: {}'.format(max_action_size))

POLICY_DEEP = [False]
VALUE_DEEP = [False]

experiment = 'progressive'

# Add transfer learning suffix
transfer_suf = ''
if experiment == 'finetune':
    transfer_suf = '_finetuned_from'
elif experiment == 'progressive':
    transfer_suf = '_progressive_from'

# Log path
PATH_LOG_LIST = ['./saved_logs/actor_critic_cp'+transfer_suf,
                 './saved_logs/actor_critic_mcc'+transfer_suf,
                 './saved_logs/actor_critic_acrobot'+transfer_suf]

# Save model path
PATH_SAVE_MODEL_LIST = ['./saved_models/actor_critic_cp'+transfer_suf,
                        './saved_models/actor_critic_mcc'+transfer_suf,
                        './saved_models/actor_critic_acrobot'+transfer_suf]

# Load model path (only simple models for our experiments)
PATH_LOAD_MODEL_LIST = ['./saved_models/actor_critic_cp',
                        './saved_models/actor_critic_mcc',
                        './saved_models/actor_critic_acrobot']

# Define hyper parameters
discount_factor = 0.99
lr_list_policy = [0.001]  # [0.001, 0.0001]
lr_list_value = [0.01, 0.005, 0.001]  # [0.01, 0.005, 0.001]

render = False

# --------------------------------------------- Run Experiment --------------------------------------------- #
envs_names_list = ['cp', 'mcc', 'acrobot']

if experiment == 'simple':
    env_idx = 2
    finetune_agent.actorCriticRunner(env=envs_list[env_idx],
                                     env_state_size=state_size_list[env_idx],
                                     env_action_size=action_size_list[env_idx],
                                     state_size=max_state_size,
                                     action_size=max_action_size,
                                     policy_deep=POLICY_DEEP,
                                     value_deep=VALUE_DEEP,
                                     lr_list_policy=lr_list_policy,
                                     lr_list_value=lr_list_value,
                                     discount_factor=discount_factor,
                                     render=render,
                                     env_name=envs_names_list[env_idx],
                                     path_log=PATH_LOG_LIST[env_idx],
                                     path_save=PATH_SAVE_MODEL_LIST[env_idx],
                                     path_load=None,
                                     fine_tune_env_name=None,
                                     convergence_amount=envs_convergence_list[env_idx],
                                     max_episodes=max_episodes_list[env_idx],
                                     max_steps=max_steps_list[env_idx],
                                     actions_continuous=actions_mcc,
                                     experiment=experiment,
                                     sources_sfx_progressive=[]
                                     )
elif experiment == 'finetune':
    env_from_idx = 0  # transfer from cp
    env_to_idx = 1  # to mcc
    path_log = PATH_LOG_LIST[env_to_idx] + '_' + envs_names_list[env_from_idx]
    path_save = PATH_SAVE_MODEL_LIST[env_to_idx] + '_' + envs_names_list[env_from_idx]

    print('Transfer learning - fine tune from {} to {}'.format(envs_names_list[env_from_idx],
                                                               envs_names_list[env_to_idx]))

    finetune_agent.actorCriticRunner(env=envs_list[env_to_idx],
                                     env_state_size=state_size_list[env_to_idx],
                                     env_action_size=action_size_list[env_to_idx],
                                     state_size=max_state_size,
                                     action_size=max_action_size,
                                     policy_deep=POLICY_DEEP,
                                     value_deep=VALUE_DEEP,
                                     lr_list_policy=lr_list_policy,
                                     lr_list_value=lr_list_value,
                                     discount_factor=discount_factor,
                                     render=render,
                                     env_name=envs_names_list[env_to_idx],
                                     path_log=path_log,
                                     path_save=path_save,
                                     path_load=PATH_LOAD_MODEL_LIST[env_from_idx],
                                     fine_tune_env_name=envs_names_list[env_from_idx],
                                     convergence_amount=envs_convergence_list[env_to_idx],
                                     max_episodes=max_episodes_list[env_to_idx],
                                     max_steps=max_steps_list[env_to_idx],
                                     actions_continuous=actions_mcc,
                                     experiment=experiment,
                                     sources_sfx_progressive=[]
                                     )
elif experiment == 'progressive':
    env_idx = 0
    sources = [2, 1]
    paths_load = [PATH_LOAD_MODEL_LIST[s_idx] for s_idx in sources]
    path_log = PATH_LOG_LIST[env_idx] + '_' + '_'.join([envs_names_list[s] for s in sources])
    path_save = PATH_SAVE_MODEL_LIST[env_idx] + '_' + '_'.join([envs_names_list[s] for s in sources])
    topology = [max_state_size, 12, 30, 8, max_action_size] if POLICY_DEEP[0] else [max_state_size, 12, max_action_size]
    target_activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu] if POLICY_DEEP[0] else [tf.nn.relu]

    sources_sfx_progressive = [envs_names_list[s_idx] for s_idx in sources]


    progressive_networks_agent.actorCriticRunner(env=envs_list[env_idx],
                                                 env_state_size=state_size_list[env_idx],
                                                 env_action_size=action_size_list[env_idx],
                                                 state_size=max_state_size,
                                                 action_size=max_action_size,
                                                 policy_deep=POLICY_DEEP,
                                                 value_deep=VALUE_DEEP,
                                                 lr_list_policy=lr_list_policy,
                                                 lr_list_value=lr_list_value,
                                                 discount_factor=discount_factor,
                                                 render=render,
                                                 env_name=envs_names_list[env_idx],
                                                 path_log=path_log,
                                                 path_save=path_save,
                                                 paths_load=paths_load,
                                                 target_activations=target_activations,
                                                 target_topology=topology,
                                                 convergence_amount=envs_convergence_list[env_idx],
                                                 max_episodes=max_episodes_list[env_idx],
                                                 max_steps=max_steps_list[env_idx],
                                                 actions_continuous=actions_mcc,
                                                 experiment=experiment,
                                                 sources_sfx_progressive=sources_sfx_progressive
                                                 )
