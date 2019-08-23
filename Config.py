import os
import random
import tensorflow as tf


checkpoint_path = os.getcwd() + '/checkpoint'
model_path = os.getcwd() + '/model/model.h5'
data_path = os.getcwd() + '/data'
frozen_path = os.getcwd() + 'frozen_model.pb'