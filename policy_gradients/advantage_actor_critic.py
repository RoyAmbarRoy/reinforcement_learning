import gym
import numpy as np
import tensorflow as tf
import collections
from policy_gradients.utils import creators
import time

env = gym.make('CartPole-v1')
np.random.seed(1)

POLICY_DEEP = [True, False]
VALUE_DEEP = [True, False]
# Log path
PATH_LOG = './saved_logs/advantage_actor_critic'

# Define hyperparameters -  Get number of inputs and outputs from environment
state_size = env.observation_space.high.size
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
lr_list_policy = [0.001, 0.0001]
lr_list_value = [0.01, 0.005, 0.001]

render = False

# Experiment: try different network structures and different learning rates
for is_deep_p in POLICY_DEEP:
    for is_deep_v in VALUE_DEEP:
        for lr_p in lr_list_policy:
            for lr_v in lr_list_value:

                # Initialize the policy network
                tf.reset_default_graph()
                # Initialize the policy and value network
                policy_network, value_network = creators.createPolicyValueNetworks(state_size,
                                                                                   action_size,
                                                                                   lr_p,
                                                                                   lr_v,
                                                                                   is_deep_p,
                                                                                   is_deep_v)
                tf.summary.merge_all()

                # Add suffixes to the saved_logs path
                p_net_deepness = ('deep' if is_deep_p else 'shallow')
                v_net_deepness = ('deep' if is_deep_v else 'shallow')
                path_log_conc = PATH_LOG + '_P{}_V{}'.format(p_net_deepness, v_net_deepness)
                path_log_conc += '/lr_p={};lr_v={}'.format(lr_p, lr_v)

                file_writer = tf.summary.FileWriter(path_log_conc, tf.get_default_graph())

                print('Using {} policy (actor) network and {} value (critic) network'.format(
                    p_net_deepness, v_net_deepness
                ))
                print('Learning rate policy = ' + str(lr_p) + ';  Learning rate value = ' + str(lr_v))

                # Start training the agent with actor-critic algorithm
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    solved = False
                    Transition = collections.namedtuple("Transition",
                                                        ["state", "action", "reward", "next_state", "done"])
                    episode_rewards = np.zeros(max_episodes)
                    average_rewards = 0.0

                    # Start timestamp
                    start_time = time.time()

                    for episode in range(max_episodes):
                        state = env.reset()
                        state = state.reshape([1, state_size])
                        episode_transitions = []
                        losses_p = []
                        losses_v = []
                        I = 1.0

                        for step in range(max_steps):
                            actions_distribution = sess.run(policy_network.actions_distribution,
                                                            {policy_network.state: state})  # predict actions
                            # Stochastically select the action - choose action with created action probabilities
                            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                            next_state, reward, done, _ = env.step(action)
                            next_state = next_state.reshape([1, state_size])

                            action_one_hot = np.zeros(action_size)
                            action_one_hot[action] = 1

                            # Calculate the TD Error
                            value_next = sess.run(value_network.state_value, {value_network.state: next_state})  # Vt+1
                            td_target = reward + discount_factor * (0 if done else value_next)  # Gt
                            value_curr = sess.run(value_network.state_value, {value_network.state: state})  # Vt
                            td_error = td_target - value_curr  # td_target - Vt

                            # Update the Critic
                            feed_dict_value_estimation = {value_network.state: state,
                                                          value_network.R_t: td_target}
                            _, value_loss = sess.run([value_network.optimizer, value_network.loss],
                                                     feed_dict_value_estimation)
                            losses_v.append(value_loss)

                            # Update the Actor and update the network's weights
                            # using the td error as our advantage estimate
                            feed_dict = {policy_network.state: state,
                                         policy_network.R_t: td_error,
                                         policy_network.action: action_one_hot}
                            _, policy_loss = sess.run([policy_network.optimizer, policy_network.loss], feed_dict)
                            losses_p.append(policy_loss)

                            if render:
                                env.render()

                            episode_transitions.append(
                                Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state,
                                           done=done))
                            episode_rewards[episode] += reward

                            if done:
                                if episode > 98:
                                    # Check if solved
                                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                                                                                                   episode_rewards[
                                                                                                       episode], round(
                                        average_rewards, 2)))
                                if average_rewards > 475:
                                    print(' Solved at episode: ' + str(episode))
                                    solved = True
                                break
                            I = discount_factor * I
                            state = next_state

                        # Write summaries
                        avg_loss_p = np.mean(losses_p)
                        avg_loss_v = np.mean(losses_v)
                        file_writer.add_summary(creators.create_summary('rewards', episode_rewards[episode]), episode)
                        file_writer.add_summary(creators.create_summary('rewards (last 100)', average_rewards), episode)
                        file_writer.add_summary(creators.create_summary('loss policy nn', avg_loss_p), episode)
                        file_writer.add_summary(creators.create_summary('loss value nn', avg_loss_v), episode)

                        if solved:
                            break

                    # End timestamp
                    end_time = time.time()
                    running_time = end_time - start_time
                    file_writer.add_summary(creators.create_summary('running_time', running_time))

                    file_writer.close()