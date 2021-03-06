"""
Testing fix len programs
"""
import numpy as np
import torch
import os
import json
from torch.autograd.variable import Variable
from src.Utils import read_config
from src.Generator.m_generator import M_Generator
from src.Models.models import ParseModelOutput
from src.Models.m_models import M_CsgNet
from src.Utils.train_utils import prepare_input_op, stack_from_expressions, beams_parser

from src.display.glm import glm
from src.projection.projection import *

import time
import sys

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config.yml")
print(config.config)


data_labels_paths = {3: "data/one_op/expressions.txt",
                     5: "data/two_ops/expressions.txt",
                     7: "data/three_ops/expressions.txt"}
dataset_sizes = {3: [110000, 1000],
                 5: [220000, 2000],
                 7: [440000, 4000]}

test_gen_objs = {}
types_prog = len(dataset_sizes.keys())
max_len = max(data_labels_paths.keys())
beam_width = 10
stack_size = max_len // 2 + 1

generator = M_Generator(data_labels_paths=data_labels_paths,
                      batch_size=config.batch_size,
                      time_steps=max(data_labels_paths.keys()),
                      stack_size=max(data_labels_paths.keys()) // 2 + 1)

imitate_net = M_CsgNet(grid_shape=[64, 64, 64], dropout=config.dropout,
                     mode=config.mode, timesteps=max_len,
                     num_draws=len(generator.unique_draw),
                     in_sz=config.input_size,
                     hd_sz=config.hidden_size,
                     stack_len=config.top_k)

cuda_devices = torch.cuda.device_count()

if torch.cuda.device_count() > 1:
    imitate_net.cuda_devices = torch.cuda.device_count()
    print("using multi gpus", flush=True)
    imitate_net = torch.nn.DataParallel(imitate_net, dim=0)
    imitate_net.load_state_dict(torch.load(config.pretrain_modelpath))
else:
    #weights = torch.load(config.pretrain_modelpath)
    #new_weights = {}
    #for k in weights.keys():
    #    if k.startswith("module"):
    #        new_weights[k[7:]] = weights[k]
    #imitate_net.load_state_dict(new_weights)

    imitate_net.load_state_dict(torch.load(config.pretrain_modelpath))

imitate_net.cuda()

for param in imitate_net.parameters():
    param.requires_grad = True

config.test_size = sum((dataset_sizes[k][1] // config.batch_size) * config.batch_size
                       for k in dataset_sizes.keys())

for k in data_labels_paths.keys():
    # if using multi gpu training, train and test batch size should be multiple of
    # number of GPU edvices.
    test_batch_size = config.batch_size
    test_gen_objs[k] = generator.get_test_data(test_batch_size,
                                               k,
                                               num_train_images=dataset_sizes[k][0],
                                               num_test_images=dataset_sizes[k][1],
                                               if_primitives=True,
                                               final_canvas=True,
                                               if_jitter=False)

Target_expressions = []
Predicted_expressions = []

parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1, max_len, [64, 64, 64], primitives=generator.primitives)
imitate_net.eval()
Rs = 0
t1 = time.time()
IOU = {}
total_iou = 0
print('begin testing')
for k in data_labels_paths.keys():
    Rs = 0.0
    for batch_idx in range(dataset_sizes[k][1] // config.batch_size):
        data_, labels = next(test_gen_objs[k])
        data_ = data_[:, :, 0:config.top_k + 1, :, :]
        one_hot_labels = prepare_input_op(labels, len(generator.unique_draw))
        one_hot_labels = Variable(torch.from_numpy(one_hot_labels), volatile=True).cuda()
        data = Variable(torch.from_numpy(data_)).cuda()
        labels = Variable(torch.from_numpy(labels)).cuda()
        # This is for data parallelism purpose
        data = data.permute(1, 0, 2, 3, 4)

        all_beams, next_beams_prob, all_inputs = imitate_net.beam_search_mode_1(
            [data, one_hot_labels], beam_width, max_len)
        beam_labels = beams_parser(all_beams, batch_size=config.batch_size,
                                   beam_width=beam_width)

        beam_labels_numpy = np.zeros((config.batch_size * beam_width, max_len),
                                     dtype=np.int32)

        for i in range(data_.shape[1]):
            beam_labels_numpy[i * beam_width: (i + 1) * beam_width, :] = beam_labels[i]

        # find expression from these predicted beam labels
        expressions = [""] * config.batch_size * beam_width
        for i in range(config.batch_size * beam_width):
            for j in range(max_len):
                expressions[i] += generator.unique_draw[beam_labels_numpy[i, j]]
        for index, prog in enumerate(expressions):
            expressions[index] = prog.split("$")[0]

        #Predicted_expressions += expressions
        target_expressions = parser.labels2exps(labels, k)
        Target_expressions += target_expressions

        target_stacks = parser.expression2stack(target_expressions)

        target_voxels = target_stacks[-1,:,0,:,:,:].astype(dtype=bool)
        target_voxels_new = np.repeat(target_voxels, axis=0,
                                      repeats=beam_width)
        predicted_stack = stack_from_expressions(parser, expressions)

        beam_R = np.sum(np.logical_and(target_voxels_new, predicted_stack), (1, 2,
                                                                             3)) / \
                 (np.sum(np.logical_or(target_voxels_new, predicted_stack), (1, 2,
                                                                             3)) + 1)
        axis = glm.vec3(1,1,1)
        transfer_matrix = axis_view_matrix(axis=axis)
        center = np.dot( transfer_matrix,np.array([32,32,32],dtype=float) )
        
        # choose an output whose projection is the most similar to the input 2D image
        predicted_images = np.zeros([beam_width * config.batch_size, 128, 128], dtype = float)
        for index,voxel in enumerate(predicted_stack):
            point_list = axis_view_place_points(voxel, transfer_matrix = transfer_matrix)

            img = z_parrallel_projection_point_simple(point_list,origin_w=128,origin_h=128, origin_z=128, w=128, h=128, center_x=center[0], center_y=center[1])
        
            predicted_images[index,:,:] = img

        target_images = data_[-1, :, 0, :, :]
        target_images_new = np.repeat(target_images, axis=0, repeats=beam_width)
        beam_choose_R = np.sum( np.abs(predicted_images - target_images_new), (1,2) )
        
        # There are some expressions as input that result in the blank canvas. So we just
        # set these input's reward to the 0 value of top_1 prediction.
        where_are_NaNs = np.isnan(beam_R)
        beam_R[where_are_NaNs] = 0.0

        R = np.zeros(config.batch_size)
        for r in range(config.batch_size):
            max_choose_R = np.min(beam_choose_R[r * beam_width: (r + 1) * beam_width])

            # find the index of the best expression
            max_index = np.where(beam_choose_R [r * beam_width: (r + 1) * beam_width] == max_choose_R)
            
            R[r] = beam_R[ max_index[0][0] + r * beam_width ]
            Predicted_expressions.append(expressions[max_index[0][0] + r * beam_width])

        Rs += np.sum(R)
        
        print('batch' + str(batch_idx))
    total_iou += Rs
    IOU[k] = Rs / ((dataset_sizes[k][1] // config.batch_size) * config.batch_size)
    print("IOU for {} len program: ".format(k), IOU[k])


total_iou = total_iou / config.test_size
print ("total IOU score: ", total_iou)
results = {"total_iou": total_iou, "iou": IOU}

results_path = "trained_models/results/{}/".format(config.pretrain_modelpath)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path + "beam_{}_pred-M.txt".format(beam_width), "w") as file:
    for p in Predicted_expressions:
        file.write(p + "\n")

with open(results_path + "beam_{}_target-M.txt".format(beam_width), "w") as file:
    for p in Target_expressions:
        file.write(p + "\n")

with open(results_path + "beam_{}_results-M.org".format(beam_width), 'w') as outfile:
    json.dump(results, outfile)
