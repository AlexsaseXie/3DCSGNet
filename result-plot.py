import numpy as np
import time

import random
import matplotlib.pyplot as plt
import matplotlib 

import sys 

if (len(sys.argv) <= 1):
    file_path = 'trained_models/results/trained_models/2D-transfer-100epoch-2.pth'
    m_path = ['/beam_10_pred.txt', '/beam_10_target.txt']
else:
    file_path = sys.argv[1]
    if (sys.argv[2] == 'test'):
        m_path = ['/pred.txt', '/target.txt']
    elif (sys.argv[2] == 'beam'):
        m_path = ['/beam_10_pred.txt', '/beam_10_target.txt']
    else: 
        m_path = ['/beam_10_pred-M.txt', '/beam_10_target-M.txt']

with open(file_path + m_path[0]) as data_file:
    expressions = data_file.readlines()
with open(file_path + m_path[1]) as target_data_file:
    target_expressions = target_data_file.readlines()


import deepdish as dd
from src.Utils.train_utils import voxels_from_expressions

primitives = dd.io.load("data/primitives.h5")

print('load primitive finish')

result_count = len(expressions)

ious = np.zeros(result_count ,dtype=float)

for i in range(result_count):
    voxels = voxels_from_expressions([expressions[i], target_expressions[i]], primitives, max_len=7)

    voxel_pred = voxels[0]
    voxel_target = voxels[1]

    iou = np.sum(np.logical_and(voxel_pred, voxel_target)) / (np.sum(
            np.logical_or(voxel_pred, voxel_target)) + 1)

    ious[i] = iou

    if (i % 100 == 0):
        print(i, ' finished!')


plt.hist(ious, bins=40, normed=1, facecolor="blue", edgecolor="black", alpha=0.7)
plt.show()


#output the index of bad preds

if (m_path[0] == '/pred.txt'):
    temp = "test_"
elif (m_path[0] == '/beam_10_pred.txt'):
    temp = "beam_"
else:
    temp = "beam-M_"

f = open(file_path + "/"+ temp + "bad_result.txt", "w")

f.write('iou < 0.6\n')

bad_preds = np.argwhere(ious < 0.6)

for index in bad_preds:
    f.write(str(index[0]) + '\n')

ious[ious < 0.6] = 100

f.write('0.6 <= iou < 0.8\n')

bad_preds = np.argwhere(ious < 0.8)

for index in bad_preds:
    f.write(str(index[0]) + '\n')

