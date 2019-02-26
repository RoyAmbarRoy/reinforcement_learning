import numpy as np
import tensorflow as tf
import transfer_learning.utils.creators_utils as c_utils
import transfer_learning.utils.model_loaders as m_loaders

def padArrZeros(arr, pad_size):
    return np.pad(arr, (0, pad_size), 'constant')

def softMax(arr):
    e_sum = sum([np.e ** a for a in arr])
    soft_max = [np.e ** a / e_sum for a in arr]
    return soft_max

def chooseAction(actions_distribution, env_action_size, do_softmax=False):
    # Ignore unnecessary actions
    actions_distribution = actions_distribution[0][0:env_action_size]
    if do_softmax:
        actions_distribution = softMax(actions_distribution)
    # Normalize the distribution (sum to 1)
    actions_distribution /= sum(actions_distribution)
    # Stochastically select the action - choose action with created action probabilities
    action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
    return action

def buildPath(initial_path, p_net_deepness, v_net_deepness, lr_p, lr_v):
    # Add suffixes to the saved_logs path
    path_conc = initial_path + '_P{}_V{}'.format(p_net_deepness, v_net_deepness)
    path_conc += '/lr_p={}_lr_v={}'.format(lr_p, lr_v)
    return path_conc

def actorCriticRunner(env,
                      env_state_size,
                      env_action_size,
                      state_size,
                      action_size,
                      policy_deep,
                      value_deep,
                      lr_list_policy,
                      lr_list_value,
                      discount_factor,
                      render,
                      env_name,
                      path_log,
                      path_save=None,
                      path_load=None,
                      networks_prefixes=None,
                      convergence_amount=475,
                      max_episodes=5000,
                      max_steps=501,
                      actions_continuous=None):
    np.random.seed(42)
    env.seed(42)
    pad_size_state = state_size - env_state_size
    pad_size_action = action_size - env_action_size

    # Experiment: try different network structures and different learning rates
    for is_deep_p in policy_deep:
        for is_deep_v in value_deep:
            for lr_p in lr_list_policy:
                for lr_v in lr_list_value:
                    # Initialize the policy network
                    tf.reset_default_graph()
                    # Initialize the policy and value network
                    policy_network, value_network = c_utils.createPolicyValueNetworks(state_size,
                                                                                      action_size,
                                                                                      lr_p,
                                                                                      lr_v,
                                                                                      is_deep_p,
                                                                                      is_deep_v)
                    # Merge summaries
                    tf.summary.merge_all()
                    # Create a saver object
                    saver = tf.train.Saver()

                    p_net_deepness = ('deep' if is_deep_p else 'shallow')
                    v_net_deepness = ('deep' if is_deep_v else 'shallow')
                    path_conc = buildPath(path_log, p_net_deepness, v_net_deepness, lr_p, lr_v)
                    file_writer = tf.summary.FileWriter(path_conc, tf.get_default_graph())

                    print('Using {} policy (actor) network and {} value (critic) network'.format(
                        p_net_deepness, v_net_deepness
                    ))
                    print('Learning rate policy = ' + str(lr_p) + ';  Learning rate value = ' + str(lr_v))
                    w4 = tf.get_default_graph().get_tensor_by_name('policy_network_deep/W4:0')
                    # Start training the agent with actor-critic algorithm
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())

                        if path_load:
                            # Load pre-trained graph models
                            path_conc = buildPath(path_load, p_net_deepness, v_net_deepness, lr_p, lr_v)
                            m_loaders.loadModel(path_conc, sess, saver)
                            # Re-initialize last layer weights
                            m_loaders.initializeLastLayer(path_conc, sess, networks_prefixes, verbose=True)

                        solved = False
                        episode_rewards = np.zeros(max_episodes)
                        episode_rewards_synth = np.zeros(max_episodes)
                        average_rewards = 0.0
                        average_rewards_synth = 0.0

                        for episode in range(max_episodes):
                            state = env.reset()
                            state = padArrZeros(state, pad_size_state)
                            state = state.reshape([1, state_size])
                            losses_p = []
                            losses_v = []
                            i = 1.0

                            for step in range(max_steps):
                                # predict actions
                                actions_distribution = sess.run(policy_network.output,
                                                                {policy_network.state: state})
                                #actions_distribution = sess.run(policy_network.actions_distribution,
                                #                                {policy_network.state: state})

                                action = chooseAction(actions_distribution, env_action_size, True)
                                action_one_hot = c_utils.createOneHot(action_size, action)

                                action = [actions_continuous[action]] if env_name == 'MCC' else action
                                next_state, reward, done, _ = env.step(action)
                                episode_rewards[episode] += reward
                                next_state = padArrZeros(next_state, pad_size_state)
                                next_state = next_state.reshape([1, state_size])

                                if env_name == 'MCC':
                                    next_vel = abs(next_state[0][1])
                                    next_pos = abs(next_state[0][0])
                                    reward = reward if done else next_vel * next_pos
                                    episode_rewards_synth[episode] += reward

                                # Calculate the TD Error
                                value_next = sess.run(value_network.state_value,
                                                      {value_network.state: next_state})  # Vt+1
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
                                _, policy_loss = sess.run([policy_network.optimizer,
                                                           policy_network.loss],
                                                          feed_dict)
                                losses_p.append(policy_loss)

                                if render:
                                    env.render()

                                if done:
                                    if episode > 98:
                                        # Check if solved
                                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                                        average_rewards_synth = np.mean(episode_rewards_synth[(episode - 99):episode + 1])
                                    print("Episode {} "
                                          "Step {} "
                                          "Reward: {} "
                                          "Synthetic Reward: {} "
                                          "Average over 100 episodes: {}".format(episode,
                                                                                 step,
                                                                                 episode_rewards[episode],
                                                                                 episode_rewards_synth[episode],
                                                                                 round(average_rewards, 2)))
                                    if episode > 98 and \
                                       average_rewards > convergence_amount:
                                        print(' Solved at episode: ' + str(episode))
                                        solved = True
                                    break
                                i = discount_factor * i
                                state = next_state

                            # Write summaries
                            avg_loss_p = np.mean(losses_p)
                            avg_loss_v = np.mean(losses_v)
                            file_writer.add_summary(c_utils.create_summary('rewards', episode_rewards[episode]),
                                                    episode)
                            file_writer.add_summary(c_utils.create_summary('rewards (last 100)',
                                                                           average_rewards),
                                                    episode)
                            file_writer.add_summary(c_utils.create_summary('rewards synthetic (last 100)',
                                                                         average_rewards_synth),
                                                    episode)
                            file_writer.add_summary(c_utils.create_summary('loss policy nn', avg_loss_p), episode)
                            file_writer.add_summary(c_utils.create_summary('loss value nn', avg_loss_v), episode)

                            if solved:
                                break

                        if path_save:
                            # Save the variables to disk.
                            path_conc = buildPath(path_save, p_net_deepness, v_net_deepness, lr_p, lr_v)
                            c_utils.createDir(path_conc)
                            saver.save(sess, path_conc+'/model')
                        file_writer.close()