import gym
import transfer_learning.runners as runners

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
envs_names_list = ['CP', 'MCC', 'ACROBOT']
state_size_list = [len(env_cp.reset()), len(env_mcc.reset()), len(env_acrobot.reset())]
action_size_list = [env_cp.action_space.n, CONTINUOUS_N_OF_BINS, env_acrobot.action_space.n]
max_state_size = max(state_size_list)
max_action_size = max(action_size_list)

envs_convergence_list = [475, 80, -100]

print('max state size: {}'.format(max_state_size))
print('max action size: {}'.format(max_action_size))

POLICY_DEEP = [True]
VALUE_DEEP = [True]

DO_TRANSFER = False
transfer_pref = '_transfered_from_' if DO_TRANSFER else ''

# Log path
PATH_LOG_LIST = ['./saved_logs/actor_critic_cp'+transfer_pref,
                 './saved_logs/actor_critic_mcc'+transfer_pref,
                 './saved_logs/actor_critic_acrobot'+transfer_pref]
# Save model path
PATH_SAVE_MODEL_LIST = ['./saved_models/actor_critic_cp'+transfer_pref,
                        './saved_models/actor_critic_mcc'+transfer_pref,
                        './saved_models/actor_critic_acrobot'+transfer_pref]
# Load model path
PATH_LOAD_MODEL_LIST = ['./saved_models/actor_critic_cp',
                        './saved_models/actor_critic_mcc',
                        './saved_models/actor_critic_acrobot']

discount_factor = 0.99
lr_list_policy = [0.001, 0.0001]
lr_list_value = [0.01, 0.005, 0.001]

render = False

if not DO_TRANSFER:
    env_idx = 2
    # Perform simple training on one environment
    runners.actorCriticRunner(env=envs_list[env_idx], env_state_size=state_size_list[env_idx],
                              env_action_size=action_size_list[env_idx],
                              state_size=max_state_size, action_size=max_action_size,
                              policy_deep=POLICY_DEEP, value_deep=VALUE_DEEP, lr_list_policy=lr_list_policy,
                              lr_list_value=lr_list_value, discount_factor=discount_factor, render=render,
                              env_name=envs_names_list[env_idx],
                              path_log=PATH_LOG_LIST[env_idx],
                              path_save=PATH_SAVE_MODEL_LIST[env_idx],
                              path_load=None,
                              networks_prefixes=None,
                              convergence_amount=envs_convergence_list[env_idx],
                              max_episodes=5000, max_steps=7000, actions_continuous=actions_mcc)

    """
    # Perform simple training over all environments
    for i in range(len(envs_list)):
        runners.actorCriticRunner(env=envs_list[2], env_state_size=state_size_list[2], env_action_size=action_size_list[2],
                                  state_size=max_state_size, action_size=max_action_size,
                                  POLICY_DEEP=POLICY_DEEP, VALUE_DEEP=VALUE_DEEP, lr_list_policy=lr_list_policy,
                                  lr_list_value=lr_list_value, discount_factor=discount_factor, render=render,
                                  PATH_LOG=PATH_LOG_LIST[2], PATH_SAVE=PATH_SAVE_MODEL_LIST[2], PATH_LOAD=None,
                                  env_name=envs_names_list[2], convergence_amount=envs_convergence_list[2],
                                  max_episodes=5000, max_steps=7000, actions_continuous=actions_mcc)
    """
else:
    networks_prefixes = ['policy_network_{}'.format('deep' if POLICY_DEEP else 'shallow'),
                         'value_network_{}'.format('deep' if VALUE_DEEP else 'shallow')]
    env_from_idx = 0  # transfer from cp
    env_to_idx = 1    # to mcc
    path_log = PATH_LOG_LIST[env_to_idx] + '_' + envs_names_list[env_from_idx].lower()
    path_save = PATH_SAVE_MODEL_LIST[env_to_idx] + '_' + envs_names_list[env_from_idx].lower()
    # Perform transfer learning from one environment to another
    runners.actorCriticRunner(env=envs_list[env_to_idx],
                              env_state_size=state_size_list[env_to_idx],
                              env_action_size=action_size_list[env_to_idx],
                              state_size=max_state_size, action_size=max_action_size,
                              policy_deep=POLICY_DEEP, value_deep=VALUE_DEEP,
                              lr_list_policy=lr_list_policy, lr_list_value=lr_list_value,
                              discount_factor=discount_factor, render=render,
                              env_name=envs_names_list[env_to_idx],
                              path_log=path_log, path_save=path_save,
                              path_load=PATH_LOAD_MODEL_LIST[env_from_idx],
                              networks_prefixes=networks_prefixes,
                              convergence_amount=envs_convergence_list[env_to_idx],
                              max_episodes=3000, max_steps=7000, actions_continuous=actions_mcc)