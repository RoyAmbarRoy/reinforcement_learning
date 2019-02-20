import gym
import numpy as np
import tensorflow as tf
import collections
import utils

env = gym.make('CartPole-v1')
np.random.seed(1)


class PolicyNetworkShallow:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network_shallow'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class PolicyNetworkDeep:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 30],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [30], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [30, 8],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [8], initializer=tf.zeros_initializer())
            self.W4 = tf.get_variable("W4", [8, self.action_size],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b4 = tf.get_variable("b4", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.Z3 = tf.add(tf.matmul(self.A2, self.W3), self.b3)
            self.A3 = tf.nn.relu(self.Z3)
            self.output = tf.add(tf.matmul(self.A3, self.W4), self.b4)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetworkShallow:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 1],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # State value output
            self.state_value = tf.squeeze(self.output)
            # Loss squared difference between v and r
            self.loss = tf.losses.mean_squared_error(self.state_value, self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetworkDeep:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 30],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [30], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [30, 8],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [8], initializer=tf.zeros_initializer())
            self.W4 = tf.get_variable("W4", [8, 1],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b4 = tf.get_variable("b4", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.Z3 = tf.add(tf.matmul(self.A2, self.W3), self.b3)
            self.A3 = tf.nn.relu(self.Z3)
            self.output = tf.add(tf.matmul(self.A3, self.W4), self.b4)

            # State value output
            self.state_value = tf.squeeze(self.output)
            # Loss squared difference between v and r
            self.loss = tf.losses.mean_squared_error(self.state_value, self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def createActorCriticNetworks(state_size, action_size, lr_p, lr_v,
                              is_deep_p, is_deep_v):
    p_n = PolicyNetworkDeep(state_size, action_size, lr_p) if is_deep_p \
        else PolicyNetworkShallow(state_size, action_size, lr_p)
    v_n = ValueNetworkDeep(state_size, lr_v) if is_deep_v \
        else ValueNetworkShallow(state_size, lr_v)
    return p_n, v_n


POLICY_DEEP = [True, False]
VALUE_DEEP = [True, False]
# Log path
PATH_LOG = './log/advantage_actor_critic___'

# Define hyperparameters -  Get number of inputs and outputs from environment
state_size = env.observation_space.high.size
action_size = env.action_space.n

max_episodes = 30
max_steps = 501
discount_factor = 0.99
lr_list_policy = [0.001, 0.0001]
lr_list_value = [0.01, 0.005, 0.001]

render = False

# Experiment: try different network structures and different learning rates
for is_deep_p in POLICY_DEEP:
    for is_deep_v in POLICY_DEEP:
        for lr_p in lr_list_policy:
            for lr_v in lr_list_value:

                # Initialize the policy network
                tf.reset_default_graph()
                # Initialize the policy and value network
                policy_network, value_network = createActorCriticNetworks(state_size,
                                                                          action_size,
                                                                          lr_p, lr_v,
                                                                          is_deep_p,
                                                                          is_deep_v)
                tf.summary.merge_all()

                # Add suffixes to the log path
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
                                         policy_network.action: action_one_hot,
                                         policy_network.learning_rate: lr_p}
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
                        file_writer.add_summary(utils.create_summary('rewards', episode_rewards[episode]), episode)
                        file_writer.add_summary(utils.create_summary('rewards (last 100)', average_rewards), episode)
                        file_writer.add_summary(utils.create_summary('loss policy nn', avg_loss_p), episode)
                        file_writer.add_summary(utils.create_summary('loss value nn', avg_loss_v), episode)

                        if solved:
                            break

                    file_writer.close()