import numpy as np
import time

import random
import matplotlib.pyplot as plt
import matplotlib 


file_path = 'trained_models/results/given-model.pth'

with open(file_path + '/beam_10_pred.txt') as data_file:
    expressions = data_file.readlines()
with open(file_path + '/beam_10_target.txt') as target_data_file:
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


plt.hist(ious, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.show()


#output the index of bad preds

f = open(file_path + "/bad_result.txt", "w")

f.write('iou < 0.6\n')

bad_preds = np.argwhere(ious < 0.6)

for index in bad_preds:
    f.write(str(index[0]) + '\n')

ious[ious < 0.6] = 100

f.write('0.6 <= iou < 0.8\n')

bad_preds = np.argwhere(ious < 0.8)

for index in bad_preds:
    f.write(str(index[0]) + '\n')

