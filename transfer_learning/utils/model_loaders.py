import tensorflow as tf
import transfer_learning.utils.creators as c_utils
import numpy as np

np.random.seed(42)

def saveModel(save_path, saver, sess):
     c_utils.createDir(save_path)
     saver.save(sess, save_path+'/model')

def loadModel(loadPath, sess, saver, var_scope=None):
    saver.restore(sess, loadPath+'/model', )

def loadModels(loadPaths, sess, saver):
    for p in loadPaths:
        loadModel(p, sess, saver)

def getModelVariables(modelPath):
    # Load all variables
    list_variables = tf.contrib.framework.list_variables(modelPath + '/model')
    return list_variables

def getModelWeights(list_variables, perfix):
    # Get list of all weights
    weights_list = [item for item in list_variables if item[0].startswith(perfix)]
    return weights_list

def initializeLastLayer(sess, networksPrefixes, verbose=False):
    ops = sess.graph.get_operations()
    w_suffix = '/W'
    for nPref in networksPrefixes:
        # print([op.name for op in ops if op.name[-3:-1] == w_suffix and op.name.startswith(nPref + '/')])
        weights_names = [op.name for op in ops if op.name[-3:-1] == w_suffix and op.name.startswith(nPref + '/')]
        last_w_name = weights_names[len(weights_names)-1]
        last_w = tf.get_default_graph().get_tensor_by_name(last_w_name + ':0')
        last_w_shape = sess.run(tf.shape(last_w))
        # Create a xavier op (special for relu activation)
        mu, sigma = 0, 0.1  # mean and standard deviation
        xavier_for_relu = np.random.normal(mu, sigma, (last_w_shape[0], last_w_shape[1])) * np.sqrt(2 / last_w_shape[0])
        assign_operation = tf.assign(last_w, tf.constant(xavier_for_relu, dtype=tf.float32))
        if verbose: print(nPref + ': Last layer weights before initializing:\n{}'.format(sess.run([last_w])))
        sess.run(assign_operation)
        if verbose: print(nPref + ': Last layer weights after initializing:\n{}\n\n'.format(sess.run([last_w])))


def getTensorsByNames(names):
    tensors_list = []
    for n in names:
        tensors_list.append(tf.get_default_graph().get_tensor_by_name(n + ':0'))
    return tensors_list


def getOperations(sess, networksPrefixes):
    ops = sess.graph.get_operations()
    print([op.name for op in ops])
    w_suffix = '/W'
    b_suffix = '/b'
    a_suffix = '/Relu'
    policy_weights, policy_biases, policy_activations = [], [], []
    value_weights, value_biases, value_activations = [], [], []
    for nPref in networksPrefixes:
        weights_names = [op.name for op in ops if op.name[-3:-1] == w_suffix and op.name.startswith(nPref + '/')]
        # Pop the first W
        # weights_names.pop(0)
        biases_names = [op.name for op in ops if op.name[-3:-1] == b_suffix and op.name.startswith(nPref + '/')]
        activations_names = [op.name for op in ops if (op.name[-7:-2] == a_suffix or op.name[-5:] == a_suffix) and
                             op.name.startswith(nPref + '/')]

        weights_nodes = getTensorsByNames(weights_names)
        biases_nodes = getTensorsByNames(biases_names)
        activations_nodes = getTensorsByNames(activations_names)
        if nPref.startswith('policy'):
            policy_weights.append(weights_nodes)
            policy_biases.append(biases_nodes)
            policy_activations.append(activations_nodes)
        else:
            value_weights.append(weights_nodes)
            value_biases.append(biases_nodes)
            value_activations.append(activations_nodes)

    return policy_weights, policy_biases, policy_activations, \
        value_weights, value_biases, value_activations

"""
# Test it
policy_network, value_network = c_utils.createPolicyValueNetworks(6,
                                                                  6,
                                                                  0.001,
                                                                  0.001,
                                                                  True,
                                                                  True,
                                                                  'policy_network_deep_0.001_acrobot',
                                                                  'value_network_deep_0.001_acrobot')

policy_network_2, value_network_2 = c_utils.createPolicyValueNetworks(6,
                                                                      6,
                                                                      0.001,
                                                                      0.001,
                                                                      True,
                                                                      True,
                                                                      'policy_network_deep_0.001_mcc',
                                                                      'value_network_deep_0.001_mcc')

#
# savedPath = './saved_models/actor_critic_mcc_Pdeep_Vdeep/lr_p=0.001_lr_v=0.01'
# savedPath2 = './saved_models/actor_critic_acrobot_Pdeep_Vdeep/lr_p=0.001_lr_v=0.01'

# def assignZeros(weight, sess):
#     w4_shape = sess.run(tf.shape(weight))
#     assign_node = tf.assign(policy_network.W4, tf.constant(np.zeros(w4_shape), dtype=tf.float32))
#     sess.run(assign_node)


# ----------------------------- step 1
# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# networksPrefixes = ['policy_network_deep_mcc', 'value_network_deep_mcc',
#                     'policy_network_deep_mcc_2', 'value_network_deep_mcc_2']
#
# getOperations(sess, networksPrefixes)


# file_writer = tf.summary.FileWriter(savedPath, tf.get_default_graph())
# Save a temp model
# saveModel(save_path=savedPath, saver=saver, sess=sess)
# file_writer.close()

# exit(0)


# ----------------------------- step 2
# Load model path (only simple models for our experiments)
PATH_LOAD_MODEL_LIST = ['../saved_models/actor_critic_cp',
                        '../saved_models/actor_critic_mcc',
                        '../saved_models/actor_critic_acrobot']
path_conc = c_utils.buildPaths(PATH_LOAD_MODEL_LIST[1:], 'deep', 'deep', 0.001, 0.001)

# Load temp model
vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy_network_deep_0.001_mcc')
vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value_network_deep_0.001_mcc')
vars_3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy_network_deep_0.001_acrobot')
vars_4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value_network_deep_0.001_acrobot')

vars_1.extend(vars_2)
vars_3.extend(vars_4)
savers = [tf.train.Saver(var_list=vars_1), tf.train.Saver(var_list=vars_3)]
sess = tf.Session()
[loadModel(path_conc[i], sess, savers[i]) for i in range(len(savers))]

exit(0)
"""





#
# with tf.Session() as sess:
#     vars = tf.contrib.framework.list_variables(savedPath)
#     new_vars = []
#     for name, shape in vars:
#      v = tf.contrib.framework.load_variable(savedPath, name)
#       # print(name)
#       if name.startswith('policy_network_deep'):
#           new_vars.append(tf.Variable(v, name=name.replace('policy_network_deep', 'policy_network_deep_mcc')))
#       else:
#           new_vars.append(tf.Variable(v, name=name.replace('value_network_deep', 'value_network_deep_mcc')))
#
#     saver = tf.train.Saver(new_vars)
#     c_utils.createDir(savedPath+'_new')
#     savedPath += '_new/model'
#     saver.save(sess, savedPath)


# src_vars = [v for v in tf.all_variables() if v.name.endswith('network_deep')]
# print('old_vars:', [v.name for v in src_vars])
# out_vars = {v.name+'_1' for v in src_vars}
# print('new_vars:', [key for key in out_vars])
# tf.train.Saver(var_list=out_vars).save(sess, savedPath)

# # Initialize last layer weights
# networksPrefixes_list = ['policy_network_shallow', 'value_network_shallow']
# initializeLastLayer(modelPath=savedPath, sess=sess, networksPrefixes=networksPrefixes_list, verbose=True)

