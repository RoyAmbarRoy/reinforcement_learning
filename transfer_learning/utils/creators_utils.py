import tensorflow as tf
import transfer_learning.models.neural_nets as nns
import os
import numpy as np

def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_summary(tag, simple_value):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=simple_value)
    return summary

def createPolicyValueNetworks(state_size, action_size, lr_p, lr_v,
                              is_deep_p, is_deep_v):
    p_n = nns.PolicyNetworkDeep(state_size, action_size, lr_p) if is_deep_p \
        else nns.PolicyNetworkShallow(state_size, action_size, lr_p)
    v_n = nns.ValueNetworkDeep(state_size, lr_v) if is_deep_v \
        else nns.ValueNetworkShallow(state_size, lr_v)
    return p_n, v_n

def createOneHot(size, hot_indx):
    one_hot = np.zeros(size)
    one_hot[hot_indx] = 1
    return one_hot