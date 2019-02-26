import gym
import numpy as np
import tensorflow as tf
import collections
import policy_gradients.utils as utils

env = gym.make('CartPole-v1')
# env._max_episode_steps = None

np.random.seed(1)

POLICY_DEEP = [True, False]
VALUE_DEEP = [True, False]

# Log path
PATH_LOG = './saved_logs/advantage_policy_gradients'

# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate = 0.01
lr_list_policy = [0.001, 0.0001]
lr_list_value = [0.0001, 0.00001]

render = False

# Experiment: try different network structures and different learning rates
for is_deep_p in POLICY_DEEP:
    for is_deep_v in POLICY_DEEP:
        for lr_p in lr_list_policy:
            for lr_v in lr_list_value:

                # Initialize the policy network
                tf.reset_default_graph()
                # Initialize the policy and value network
                policy_network, value_network = utils.createPolicyValueNetworks(state_size,
                                                                               action_size,
                                                                               lr_p,
                                                                               lr_v,
                                                                               is_deep_p,
                                                                               is_deep_v)
                tf.summary.merge_all()

                # Add suffixes to the saved_logs path
                p_net_deepness = ('deep' if is_deep_p else 'shallow')
                v_net_deepness = ('deep' if is_deep_v else 'shallow')
                path_log_conc = PATH_LOG+'_P{}_V{}'.format(p_net_deepness, v_net_deepness)
                path_log_conc += '/lr_p={};lr_v={}'.format(lr_p, lr_v)

                file_writer = tf.summary.FileWriter(path_log_conc, tf.get_default_graph())

                print('Using {} policy network and {} value network'.format(
                    p_net_deepness, v_net_deepness
                ))
                print('Learning rate policy = ' + str(lr_p) + ';  Learning rate value = ' + str(lr_v))

                # Start training the agent with REINFORCE algorithm
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    solved = False
                    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
                    episode_rewards = np.zeros(max_episodes)
                    average_rewards = 0.0
                    avg_100_steps = []

                    for episode in range(max_episodes):
                        state = env.reset()
                        state = state.reshape([1, state_size])
                        episode_transitions = []

                        for step in range(max_steps):
                            actions_distribution = sess.run(policy_network.actions_distribution, {policy_network.state: state})
                            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                            next_state, reward, done, _ = env.step(action)
                            next_state = next_state.reshape([1, state_size])

                            if render:
                                env.render()

                            action_one_hot = np.zeros(action_size)
                            action_one_hot[action] = 1
                            episode_transitions.append(
                                Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                            episode_rewards[episode] += reward

                            if done:
                                if episode > 98:
                                    # Check if solved
                                    with tf.name_scope("avg_reward_over_100_steps"):
                                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                                    tf.summary.scalar('avg_reward_over_100_steps', average_rewards)
                                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                                                                                                   episode_rewards[episode],
                                                                                                   round(average_rewards, 2)))
                                avg_100_steps.append(average_rewards)
                                if average_rewards > 475:
                                    print(' Solved at episode: ' + str(episode))
                                    solved = True
                                break
                            state = next_state

                        losses_p = []
                        losses_v = []
                        # Compute Rt for each time-step t and update both networks' weights
                        for t, transition in enumerate(episode_transitions):
                            total_discounted_return = sum(
                                discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
                            # Calculate the advantage
                            baseline = sess.run(value_network.state_value, {value_network.state: state})
                            advantge = total_discounted_return - baseline
                            # Update the value network
                            feed_dict = {value_network.state: transition.state, value_network.R_t: total_discounted_return}
                            _, value_loss = sess.run([value_network.optimizer, value_network.loss], feed_dict)
                            # Update the policy network (using the advantge as the R_t)
                            feed_dict = {policy_network.state: transition.state, policy_network.R_t: advantge,
                                         policy_network.action: transition.action}
                            _, policy_loss = sess.run([policy_network.optimizer, policy_network.loss],
                                                                   feed_dict)
                            losses_v.append(value_loss)
                            losses_p.append(policy_loss)

                        # Write summaries
                        avg_loss_p = np.mean(losses_p)
                        avg_loss_v = np.mean(losses_v)
                        file_writer.add_summary(utils.create_summary('rewards', episode_rewards[episode]), episode)
                        file_writer.add_summary(utils.create_summary('rewards (last 100)', average_rewards), episode)
                        file_writer.add_summary(utils.create_summary('loss policy nn', avg_loss_p), episode)
                        file_writer.add_summary(utils.create_summary('loss value nn', avg_loss_v), episode)

                        if solved:
                            break

                file_writer.close()