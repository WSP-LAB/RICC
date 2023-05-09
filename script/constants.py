"""
'config.py' is for the global variables
"""
import os

# global variables for 'RICC.py'
base_path = os.getenv("HOME")

result_path = base_path + '/result'
dataset_path = '../dataset'
score_path = base_path + '/result/score'

FN_nodes = []
target_list = []

num_negative = 0
num_positive = 0
num_unlabel = 0
num_new_nodes = 60
node_num = 0

turn = 0
