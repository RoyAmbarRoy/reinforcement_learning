import numpy as np
import tensorflow as tf

def create_summary(tag, simple_value):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=simple_value)
    return summary


