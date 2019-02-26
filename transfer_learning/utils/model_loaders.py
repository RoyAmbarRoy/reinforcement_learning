import tensorflow as tf
import transfer_learning.utils.creators_utils as c_utils
import numpy as np

np.random.seed(42)

# def saveModel(save_path):
#     init_op = tf.initialize_all_variables()
#     saver = tf.train.Saver()
#     sess = tf.Session()
#     sess.run(init_op)
#     c_utils.createDir(save_path)
#     saver.save(sess, save_path+'/model')

def loadModel(loadPath, sess, saver):
    saver.restore(sess, loadPath+'/model')

def initializeLastLayer(modelPath, sess, networksPrefixes, verbose=False):
    # Load all variables
    list_variables = tf.contrib.framework.list_variables(modelPath+'/model')

    for netP in networksPrefixes:
        # Get list of all weights
        weights_list = [item for item in list_variables if item[0].startswith(netP+'/W')]
        # Get the shape of the last layer weights
        final_w_shape = weights_list[len(weights_list)-1][1]
        # Get the last layer weights as tensor (final_w)
        final_w_name = netP+'/W{}:0'.format(len(weights_list))
        final_w = tf.get_default_graph().get_tensor_by_name(final_w_name)

        if verbose: print(netP+': Last layer weights before initializing:\n{}'.format(sess.run([final_w])))
        # Create an initializing node connected to the last layer weights
        assign_node = tf.assign(final_w, tf.constant(np.random.rand(final_w_shape), dtype=tf.float32))
        # assign_node = tf.assign(final_w, final_w_shape, tf.contrib.layers.xavier_initializer(seed=0))
        # assign_node = tf.assign(final_w, tf.constant(np.zeros(final_w_shape), dtype=tf.float32))
        # Perform weights initialization
        sess.run(assign_node)
        if verbose: print(netP+': Last layer weights after initializing:\n{}\n\n'.format(sess.run([final_w])))





# # Test it
# policy_network, value_network = c_utils.createPolicyValueNetworks(4,
#                                                                   4,
#                                                                   0.001,
#                                                                   0.001,
#                                                                   False,
#                                                                   False)
#
# savedPath = './tmp_models/actor_critic_mcc_Pshallow_Vshallow/lr_p=0.001_lr_v=0.01'
#
# # # Save a temp model
# # saveModel(save_path=savedPath)
#
# # Load temp model
# saver = tf.train.Saver()
# sess = tf.Session()
# loadModel(loadPath=savedPath, sess=sess, saver=saver)
#
# # Initialize last layer weights
# networksPrefixes_list = ['policy_network_shallow', 'value_network_shallow']
# initializeLastLayer(modelPath=savedPath, sess=sess, networksPrefixes=networksPrefixes_list, verbose=True)
#
# exit(0)

