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
                              is_deep_p, is_deep_v, name_p, name_v):
    p_n = nns.PolicyNetworkDeep(state_size, action_size, lr_p, name_p) if is_deep_p \
        else nns.PolicyNetworkShallow(state_size, action_size, lr_p, name_p)
    v_n = nns.ValueNetworkDeep(state_size, lr_v, name_v) if is_deep_v \
        else nns.ValueNetworkShallow(state_size, lr_v, name_v)
    return p_n, v_n


def createOneHot(size, hot_indx):
    one_hot = np.zeros(size)
    one_hot[hot_indx] = 1
    return one_hot

def buildPath(initial_path, p_net_deepness, v_net_deepness, lr_p, lr_v):
    # Add suffixes to the saved_logs path
    path_conc = initial_path + '_P{}_V{}'.format(p_net_deepness, v_net_deepness)
    path_conc += '/lr_p={}_lr_v={}'.format(lr_p, lr_v)
    return path_conc

def buildPaths(initial_paths, p_net_deepness, v_net_deepness, lr_p, lr_v):
    # Add suffixes to the saved_logs path
    paths_conc = []
    for p in initial_paths:
        path_conc = p + '_P{}_V{}'.format(p_net_deepness, v_net_deepness)
        path_conc += '/lr_p={}_lr_v={}'.format(lr_p, lr_v)
        paths_conc.append(path_conc)
    return paths_conc

def deepnessSuffixes(is_deep_p, is_deep_v):
    p_net_deepness = ('deep' if is_deep_p else 'shallow')
    v_net_deepness = ('deep' if is_deep_v else 'shallow')
    return p_net_deepness, v_net_deepness

def structureSuffixes(is_deep_p, is_deep_v, lr_p, lr_v):
    p_net_deepness, v_net_deepness = deepnessSuffixes(is_deep_p, is_deep_v)
    p_net_suf = p_net_deepness+'_{}'.format(lr_p)
    v_net_suf = v_net_deepness+'_{}'.format(lr_v)
    return p_net_suf, v_net_suf

def createNetworksNames(is_deep_p, is_deep_v, lr_p, lr_v, target_name, source_names=[]):
    p_net_deepness, v_net_deepness = structureSuffixes(is_deep_p, is_deep_v, lr_p, lr_v)
    policy_pref = 'policy_network_' + p_net_deepness
    value_pref = 'value_network_' + v_net_deepness
    policy_nets_names = [policy_pref+'_'+target_name]
    value_nets_names = [value_pref+'_'+target_name]
    for name in source_names:
        policy_nets_names.append(policy_pref+'_'+name)
        value_nets_names.append(value_pref + '_' + name)

    return policy_nets_names, value_nets_names