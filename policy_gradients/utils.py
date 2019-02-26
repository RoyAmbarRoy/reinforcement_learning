import numpy as np
import tensorflow as tf
import policy_gradients.neural_nets as nns

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