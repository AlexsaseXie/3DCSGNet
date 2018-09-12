"""
Train the network using mixture of programs.
"""
import sys
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import configure, log_value
from torch.autograd.variable import Variable
from src.Utils import read_config
from src.Generator.m_generator import M_Generator
from src.Models.loss import losses_joint
from src.Models.m_models import M_CsgNet
from src.Utils.learn_utils import LearningRate
from src.Utils.train_utils import prepare_input_op, Callbacks

import time

if len(sys.argv) > 1:
    config = read_config.Config(sys.argv[1])
else:
    config = read_config.Config("config.yml")

model_name = config.model_path.format(config.proportion,
                                      config.top_k,
                                      config.hidden_size,
                                      config.batch_size,
                                      config.optim, config.lr,
                                      config.weight_decay,
                                      config.dropout,
                                      "mix",
                                      config.mode)
print(config.config)

config.write_config("log/configs/{}_config.json".format(model_name))
configure("log/tensorboard/{}".format(model_name), flush_secs=5)


callback = Callbacks(config.batch_size, "log/db/{}".format(model_name))
callback.add_element(["train_loss", "test_loss", "train_mse", "test_mse"])

data_labels_paths = {3: "data/one_op/expressions.txt",
                     5: "data/two_ops/expressions.txt",
                     7: "data/three_ops/expressions.txt"}

proportion = config.proportion  # proportion is in percentage. vary from [1, 100].

# First is training size and second is validation size per program length
dataset_sizes = {3: [proportion * 1000, proportion * 250],
                 5: [proportion * 2000, proportion * 500],
                 7: [proportion * 4000, proportion * 100]}

config.train_size = sum(dataset_sizes[k][0] for k in dataset_sizes.keys())
config.test_size = sum(dataset_sizes[k][1] for k in dataset_sizes.keys())
types_prog = len(dataset_sizes)

generator = M_Generator(data_labels_paths=data_labels_paths,
                      batch_size=config.batch_size,
                      time_steps=max(data_labels_paths.keys()),
                      stack_size=max(data_labels_paths.keys()) // 2 + 1)

imitate_net = M_CsgNet(grid_shape=[128, 128], dropout=config.dropout,
                     mode=config.mode, timesteps=max(data_labels_paths.keys()),
                     num_draws=len(generator.unique_draw),
                     in_sz=config.input_size,
                     hd_sz=config.hidden_size,
                     stack_len=config.top_k)

print(imitate_net)

# If you want to use multiple GPUs for training.
cuda_devices = torch.cuda.device_count()
if torch.cuda.device_count() > 1:
    imitate_net.cuda_devices = torch.cuda.device_count()
    print("using multi gpus", flush=True)
    imitate_net = torch.nn.DataParallel(imitate_net, device_ids=[0, 1], dim=0)
imitate_net.cuda()

if config.preload_model:
    weights = torch.load(config.pretrain_modelpath)
    new_weights = {}
    for k in weights.keys():
        if k.startswith("module"):
            new_weights[k[7:]] = weights[k]
    imitate_net.load_state_dict(new_weights)

    #imitate_net.load_state_dict(torch.load(config.pretrain_modelpath))

for param in imitate_net.parameters():
    param.requires_grad = True

if config.optim == "sgd":
    optimizer = optim.SGD(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        momentum=0.9, lr=config.lr, nesterov=False)

elif config.optim == "adam":
    optimizer = optim.Adam(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay, lr=config.lr)

reduce_plat = LearningRate(optimizer, init_lr=config.lr, lr_dacay_fact=0.2,
                           lr_decay_epoch=3, patience=config.patience)

train_gen_objs = {}

# Prefetching minibatches
for k in data_labels_paths.keys():
    # if using multi gpu training, train and test batch size should be multiple of
    # number of GPU edvices.
    train_batch_size = config.batch_size // types_prog
    test_batch_size = config.batch_size // types_prog
    train_gen_objs[k] = generator.get_train_data(train_batch_size,
                                                 k,
                                                 num_train_images=dataset_sizes[k][0],
                                                 if_primitives=True,
                                                 final_canvas=True,
                                                 if_jitter=False)

prev_test_loss = 1e20
prev_test_reward = 0
test_size = config.test_size
batch_size = config.batch_size
for epoch in range(0, config.epochs):
    train_loss = 0
    Accuracies = []
    imitate_net.train()
    # Number of times to accumulate gradients
    num_accums = config.num_traj
    for batch_idx in range(config.train_size // (config.batch_size * config.num_traj)):
        optimizer.zero_grad()
        loss_sum = Variable(torch.zeros(1)).cuda().data
        for _ in range(num_accums):
            for k in data_labels_paths.keys():
                tick = time.time()
                
                data, labels = next(train_gen_objs[k])
                
                print('fetch data cost ' + str(time.time() - tick) + 'sec')
                tick = time.time()

                data = data[:, :, 0:config.top_k + 1, :, :]
                one_hot_labels = prepare_input_op(labels, len(generator.unique_draw))
                one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
                data = Variable(torch.from_numpy(data)).cuda()
                labels = Variable(torch.from_numpy(labels)).cuda()
                data = data.permute(1, 0, 2, 3, 4)

                # forward pass
                outputs = imitate_net([data, one_hot_labels, k])

                loss = losses_joint(outputs, labels, time_steps=k + 1) / types_prog / \
                       num_accums
                loss.backward()
                loss_sum += loss.data

                print('train one batch cost' + str(time.time() - tick) + 'sec')

        # Clip the gradient to fixed value to stabilize training.
        torch.nn.utils.clip_grad_norm(imitate_net.parameters(), 20)
        optimizer.step()
        l = loss_sum
        train_loss += l
        log_value('train_loss_batch', l.cpu().numpy(), epoch * (
            config.train_size //
            (config.batch_size * num_accums)) + batch_idx)

        print('train_loss_batch @ batch' + str(epoch * (
            config.train_size //
            (config.batch_size * num_accums)) + batch_idx) + ':' , l.cpu().numpy())
        
    mean_train_loss = train_loss / (config.train_size // (config.batch_size * num_accums))
    log_value('train_loss', mean_train_loss.cpu().numpy(), epoch)
    del data, loss, loss_sum, train_loss, outputs

    print('finish epoch ' + str(epoch))
    #if test_reward > prev_test_reward:
    torch.save(imitate_net.state_dict(),
                "trained_models/{}.pth".format(model_name))
    #prev_test_reward = test_reward


    callback.dump_all()
