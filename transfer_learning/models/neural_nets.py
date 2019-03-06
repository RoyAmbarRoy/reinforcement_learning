import tensorflow as tf

class PolicyNetworkShallow:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network_shallow'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

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
            # Loss with negative saved_logs probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class PolicyNetworkDeep:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network_deep'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
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

            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))

            # Loss with negative saved_logs probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetworkShallow:
    def __init__(self, state_size, learning_rate, name='value_network_shallow'):
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
    def __init__(self, state_size, learning_rate, name='value_network_deep'):
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


# ---------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ Progressive Networks ------------------------------------------ #
# ---------------------------------------------------------------------------------------------------------- #
def getVariableShape(var, sess):
    return sess.run(tf.shape(var))


def getVariableValues(var, sess):
    return sess.run(var)


class ProgressiveColumn(object):
    def __init__(self, input_h, sess, activations,
                 weights_variables=[], biases_variables=[],
                 topology=[], name='progressive_column'):
        with tf.variable_scope(name):
            source_col = len(weights_variables) > 0
            self.length = len(weights_variables) if source_col else len(topology)-1
            self.w_variables = []
            self.b_variables = []
            self.z_ops = []
            self.h_ops = [input_h]
            self.z_ops_prog = []
            self.h_ops_prog = []
            self.activations = activations

            for i in range(self.length):
                if source_col:
                    var_w, var_b = weights_variables[i], biases_variables[i]
                    var_w_shape, var_b_shape = getVariableShape(var_w, sess), getVariableShape(var_b, sess)
                    var_w_values, var_b_values = getVariableValues(var_w, sess), getVariableValues(var_b, sess)

                    var_w_init = tf.get_variable("W{}".format(i + 1),
                                                 initializer=tf.constant(var_w_values, shape=var_w_shape))
                    var_b_init = tf.get_variable("b{}".format(i + 1),
                                                 initializer=tf.constant(var_b_values, shape=var_b_shape))

                else:
                    var_w_init = tf.get_variable("W{}".format(i + 1), [topology[i], topology[i + 1]],
                                                 initializer=tf.contrib.layers.xavier_initializer(seed=0))
                    var_b_init = tf.get_variable("b{}".format(i + 1), [topology[i + 1]],
                                                 initializer=tf.zeros_initializer())

                # Append weights, biases and activations (stopping gradient)
                self.w_variables.append(var_w_init)
                self.b_variables.append(var_b_init)
                # Activations
                z = tf.add(tf.matmul(self.h_ops[-1], var_w_init), var_b_init)
                self.z_ops.append(z)
                if i < self.length-1:
                    h = activations[i](z)
                    self.h_ops.append(h)


class ProgressiveNetwork(object):
    def __init__(self, cols_w, cols_b, cols_h, state_size,
                 sess, activations, topology, learning_rate, is_policy=True,
                 action_size=None, name='progressive_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        with tf.variable_scope(name):

            # Define one input for the entire network
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            # Define columns width and network width (columns width + 1)
            self.cols_width = len(cols_w)
            self.width = self.cols_width + 1

            # Create progressive columns list
            self.prog_cols = []
            for i in range(len(cols_w)):
                prog_col = ProgressiveColumn(self.state, sess, cols_h[i], cols_w[i], cols_b[i],
                                             name=name+'_col{}'.format(i+1))
                self.prog_cols.append(prog_col)

            # Generate the last (target) randomized column
            self.last_col_name_scope = name + '_col{}'.format(self.width)
            last_prog_col = ProgressiveColumn(self.state, sess, activations,
                                              topology=topology, name=self.last_col_name_scope)
            self.prog_cols.append(last_prog_col)

            for j in range(self.width-1):
                L = self.prog_cols[j].length
                M = self.prog_cols[j + 1].length
                i_k = 0

                # Merge columns
                for i_k, i_j in zip(reversed(range(1, M)), reversed(range(1, L))):
                    # Get h_i-1, w_i & b_i from the k'th column
                    h_k_prev = self.prog_cols[j + 1].h_ops[i_k]
                    w_k = self.prog_cols[j + 1].w_variables[i_k]
                    b_k = self.prog_cols[j + 1].b_variables[i_k]
                    Z_k_tmp = tf.add(tf.matmul(h_k_prev, w_k), b_k)

                    # Get h_i-1, w_i & b_i from the j'th column
                    h_j_prev = self.prog_cols[j].h_ops[i_j]
                    w_j = self.prog_cols[j].w_variables[i_j]
                    b_j = self.prog_cols[j].b_variables[i_j]
                    z_j_tmp = tf.add(tf.matmul(h_j_prev, w_j), b_j)

                    z_k = tf.add(Z_k_tmp, z_j_tmp)
                    h_k = self.prog_cols[j + 1].activations[i_k - 1](z_k)
                    self.prog_cols[j + 1].z_ops_prog.insert(0, z_k)
                    self.prog_cols[j + 1].h_ops_prog.insert(0, h_k)

                # Fill in the rest of the k'th column layer
                for left_layer in reversed(range(i_k)):
                    z_op = self.prog_cols[j + 1].z_ops[left_layer]
                    h_op = self.prog_cols[j + 1].h_ops[left_layer]
                    self.prog_cols[j + 1].z_ops_prog.insert(0, z_op)
                    self.prog_cols[j + 1].h_ops_prog.insert(0, h_op)

            # Generate optimizers
            M = self.prog_cols[self.width - 1].length - 1
            self.output = self.prog_cols[self.width - 1].z_ops_prog[M]
            if is_policy:
                self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
                # Loss with negative saved_logs probability
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            else:
                # State value output
                self.state_value = tf.squeeze(self.output)
                # Loss squared difference between v and r
                self.loss = tf.losses.mean_squared_error(self.state_value, self.R_t)

            # Generate optimizer for the last column (train only last column variables)
            trainable_vars = tf.trainable_variables(name+'/'+self.last_col_name_scope)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
                minimize(self.loss, var_list=trainable_vars)

