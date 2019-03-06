import numpy as np
import tensorflow as tf
import transfer_learning.utils.creators as c_utils
import transfer_learning.utils.model_loaders as m_loaders
from transfer_learning.models import neural_nets
import time

def padArrZeros(arr, pad_size):
    return np.pad(arr, (0, pad_size), 'constant')

def softMax(arr):
    e_sum = sum([np.e ** a for a in arr])
    soft_max = [np.e ** a / e_sum for a in arr]
    return soft_max

def chooseAction(actions_distribution, env_action_size, do_softmax=False):
    if do_softmax:
        # Ignore unnecessary actions
        actions_distribution = actions_distribution[0][0:env_action_size]
        actions_distribution = softMax(actions_distribution)
    else:
        # Ignore unnecessary actions
        actions_distribution = actions_distribution[0:env_action_size]

    # Normalize the distribution (sum to 1)
    actions_distribution /= sum(actions_distribution)
    # Stochastically select the action - choose action with created action probabilities
    action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
    return action

def getNetworks(state_size, action_size,
                lr_p, lr_v, is_deep_p, is_deep_v,
                policy_nets_names, value_nets_names):
    policy_networks_list, value_networks_list = [], []
    for i in range(len(policy_nets_names)):
        name_p = policy_nets_names[i]
        name_v = value_nets_names[i]
        policy_network, value_network = c_utils.createPolicyValueNetworks(state_size,
                                                                          action_size,
                                                                          lr_p,
                                                                          lr_v,
                                                                          is_deep_p,
                                                                          is_deep_v,
                                                                          name_p,
                                                                          name_v)
        policy_networks_list.append(policy_network)
        value_networks_list.append(value_network)
    return policy_networks_list, value_networks_list

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
                      paths_load,
                      target_activations,
                      target_topology,
                      path_save=None,
                      convergence_amount=475,
                      max_episodes=5000,
                      max_steps=501,
                      actions_continuous=None,
                      experiment='simple',
                      sources_sfx_progressive=[]):
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

                    policy_nets_names, value_nets_names = c_utils.createNetworksNames(is_deep_p,
                                                                                      is_deep_v,
                                                                                      lr_p,
                                                                                      lr_v,
                                                                                      env_name,
                                                                                      source_names=sources_sfx_progressive)

                    # Initialize the policy and value networks (only sources networks)
                    target_policy_name, target_value_name = policy_nets_names.pop(0), value_nets_names.pop(0)
                    policy_nets_tmp, value_nets_tmp = getNetworks(state_size,
                                                                  action_size,
                                                                  lr_p,
                                                                  lr_v,
                                                                  is_deep_p,
                                                                  is_deep_v,
                                                                  policy_nets_names,
                                                                  value_nets_names)

                    # Create savers for all networks (var scopes)
                    savers = []
                    for i in range(len(policy_nets_names)):
                        collector = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=policy_nets_names[i])
                        collector.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=value_nets_names[i]))
                        savers.append(tf.train.Saver(var_list=collector))

                    # Merge summaries
                    tf.summary.merge_all()
                    # Create a saver object
                    saver = tf.train.Saver()

                    # Generate new file writer log in the specified path
                    p_net_deepness, v_net_deepness = c_utils.deepnessSuffixes(is_deep_p, is_deep_v)
                    path_conc = c_utils.buildPath(path_log, p_net_deepness, v_net_deepness, lr_p, lr_v)
                    file_writer = tf.summary.FileWriter(path_conc, tf.get_default_graph())

                    print('Using {} policy (actor) network and {} value (critic) network'.format(
                        p_net_deepness, v_net_deepness
                    ))
                    print('Learning rate policy = ' + str(lr_p) + ';  Learning rate value = ' + str(lr_v))

                    # Start training the agent with actor-critic algorithm
                    with tf.Session() as sess:
                        # Load pre-trained graph models
                        paths_concs = c_utils.buildPaths(paths_load, p_net_deepness, v_net_deepness, lr_p, lr_v)
                        [m_loaders.loadModel(paths_concs[i], sess, savers[i]) for i in range(len(savers))]

                        networks_names = policy_nets_names + value_nets_names
                        policy_weights, policy_biases, policy_tmp_activations, \
                            value_weights, value_biases, value_tmp_activations = \
                            m_loaders.getOperations(sess, networks_names)

                        policy_activations, value_activations = [], []
                        for i in range(len(policy_tmp_activations)):
                            ops = []
                            for j in range(len(policy_tmp_activations[i])):
                                ops.append(tf.nn.relu)
                            policy_activations.append(ops)
                            value_activations.append(ops)


                        # Generate policy progressive networks
                        policy_network = neural_nets.ProgressiveNetwork(cols_w=policy_weights,
                                                                        cols_b=policy_biases,
                                                                        cols_h=policy_activations,
                                                                        state_size=state_size,
                                                                        sess=sess,
                                                                        activations=target_activations,
                                                                        topology=target_topology,
                                                                        learning_rate=lr_p,
                                                                        is_policy=True,
                                                                        action_size=action_size,
                                                                        name=target_policy_name+'_progressive_policy')

                        # Generate value progressive networks
                        value_network = neural_nets.ProgressiveNetwork(cols_w=value_weights,
                                                                       cols_b=value_biases,
                                                                       cols_h=value_activations,
                                                                       state_size=state_size,
                                                                       sess=sess,
                                                                       activations=target_activations,
                                                                       topology=target_topology,
                                                                       learning_rate=lr_v,
                                                                       is_policy=False,
                                                                       action_size=None,
                                                                       name=target_value_name+'_progressive_value')

                        sess.run(tf.global_variables_initializer())


                        solved = False
                        episode_rewards = np.zeros(max_episodes)
                        episode_rewards_synth = np.zeros(max_episodes)
                        average_rewards = 0.0
                        average_rewards_synth = 0.0

                        # Start timestamp
                        start_time = time.time()

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

                                action = [actions_continuous[action]] if env_name == 'mcc' else action
                                next_state, reward, done, _ = env.step(action)
                                episode_rewards[episode] += reward
                                next_state = padArrZeros(next_state, pad_size_state)
                                next_state = next_state.reshape([1, state_size])

                                if env_name == 'mcc':
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
                            file_writer.add_summary(c_utils.create_summary('rewards',
                                                                           episode_rewards[episode]), episode)
                            file_writer.add_summary(c_utils.create_summary('rewards (last 100)',
                                                                           average_rewards), episode)
                            file_writer.add_summary(c_utils.create_summary('rewards synthetic (last 100)',
                                                                           average_rewards_synth), episode)
                            file_writer.add_summary(c_utils.create_summary('loss policy nn', avg_loss_p), episode)
                            file_writer.add_summary(c_utils.create_summary('loss value nn', avg_loss_v), episode)

                            if solved:
                                break

                        # End timestamp
                        end_time = time.time()
                        running_time = end_time - start_time
                        file_writer.add_summary(c_utils.create_summary('running_time', running_time))

                        # print('\nActor-Critic last w after train and before save\n')
                        # print(sess.run(policy_network.W4))
                        # print(sess.run(value_network.W4))

                        if path_save:
                            # Save the variables to disk.
                            path_conc = c_utils.buildPath(path_save, p_net_deepness, v_net_deepness, lr_p, lr_v)
                            c_utils.createDir(path_conc)
                            saver.save(sess, path_conc+'/model')

                        file_writer.close()
