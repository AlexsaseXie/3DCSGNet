import numpy as np
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from ..Generator.parser import Parser
from ..Generator.stack import SimulateStack


class M_CsgNet(nn.Module):
    def __init__(self,
                 grid_shape=[128, 128],
                 dropout=0.5,
                 mode=1,
                 timesteps=3,
                 num_draws=400,
                 in_sz=2048,
                 hd_sz=2048,
                 stack_len=1):
        """
        This defines network architectures for CSG learning.
        :param dropout: Dropout to be used in non recurrent outputs of RNN
        :param mode: mode of training
        :param timesteps: Number of time steps in RNN
        :param num_draws: Number of unique primitives in the dataset
        :param in_sz: input size of features from encoder
        :param hd_sz: hidden size of RNN
        :param stack_len: Number of stack elements as input
        :param grid_shape: 3D grid structure.
        """
        super(M_CsgNet, self).__init__()

        self.input_channels = stack_len + 1
        self.in_sz = in_sz
        self.hd_sz = hd_sz
        self.num_draws = num_draws
        self.mode = mode
        self.time_steps = timesteps
        self.grid_shape = grid_shape
        self.rnn_layers = 1

        # 128 * 128 bitmap

        # Encoder architecture
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=32,
                               kernel_size=4, stride=(1, 1), padding=(2, 2))

        self.b1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=(1, 1), padding=(2, 2))

        self.b2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=(1, 1), padding=(2, 2))

        self.b3 = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=4, stride=(1, 1), padding=(2, 2))

        self.b4 = nn.BatchNorm2d(num_features=128)

        # this sequential module is created for multi gpu training.
        self._encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(dropout),
            self.b1,
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(dropout),
            self.b2,
            self.conv3,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(dropout),
            self.b3,
            self.conv4,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),
            nn.Dropout(dropout),
            self.b4,
        )

        # output 4 * 4 * 128 = 2048 

        # RNN architecture
        if (self.mode == 1) or (self.mode == 3):
            # Teacher forcing architecture, increased from previous value of 128
            self.input_op_sz = 128
            self.dense_input_op = nn.Linear(in_features=self.num_draws + 1,
                                            out_features=self.input_op_sz)

            self.rnn = nn.GRU(input_size=self.in_sz + self.input_op_sz,
                              hidden_size=self.hd_sz,
                              num_layers=self.rnn_layers,
                              batch_first=False)
            self.dense_output = nn.Linear(in_features=self.hd_sz, out_features=(
                self.num_draws))

        self.dense_fc_1 = nn.Linear(in_features=self.hd_sz,
                                    out_features=self.hd_sz)
        self.batchnorm_fc_1 = nn.BatchNorm1d(self.hd_sz, affine=False)

        self.pytorch_version = torch.__version__[2]
        if self.pytorch_version == "3":
            self.logsoftmax = nn.LogSoftmax(1)
        else:
            self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        """
        Only defines the forward pass of the encoder that takes the input 
        voxel Tensor and gives a fixed dim feature vector.
        :param x: Input tensor containing the raw voxels
        :return: fixed dim feature vector
        """
        batch_size = x.size()[0]
        x = self._encoder(x)
        x = x.view(1, batch_size, self.in_sz)
        return x

    def forward(self, x):
        """
        Defines the forward pass for the network
        :param x: This will contain data based on the type of training that 
        you do.
        :return: outputs of the network, depending upon the architecture 
        """
        if self.mode == 1:
            """Teacher forcing network"""
            data, input_op, program_len = x
            data = data.permute(1, 0, 2, 3, 4)
            batch_size = data.size()[1]
            h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
            x_f = self.encoder(data[-1, :, 0:1, :, :])
            outputs = []
            for timestep in range(0, program_len + 1):
                # X_f is always input to the network at every time step
                # along with previous predicted label
                input_op_rnn = self.relu(self.dense_input_op(input_op[:,
                                                             timestep, :]))
                input_op_rnn = input_op_rnn.unsqueeze(0)
                input = torch.cat((self.drop(x_f), input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                output = self.logsoftmax(self.dense_output(self.drop(hd)), )
                outputs.append(output)
            return outputs    


    def test(self, data):
        """ Describes test behaviour of different models"""
        if self.mode == 1:
            """Testing for teacher forcing"""
            data, input_op, program_len = data

            # This permute is used for multi gpu training, where first dimension is
            # considered as batch dimension.
            data = data.permute(1, 0, 2, 3, 4)
            batch_size = data.size()[1]
            h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
            x_f = self.encoder(data[-1, :, 0:1, :, :])
            last_op = input_op[:, 0, :]
            outputs = []
            for timestep in range(0, program_len):
                # X_f is always input to the network at every time step
                # along with previous predicted label
                input_op_rnn = self.relu(self.dense_input_op(last_op))
                input_op_rnn = input_op_rnn.unsqueeze(0)
                input = torch.cat((self.drop(x_f), input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                output = self.logsoftmax(self.dense_output(self.drop(hd)))
                outputs.append(output)
                next_input_op = torch.max(output, 1)[1]
                arr = Variable(torch.zeros(batch_size, self.num_draws + 1).scatter_(1,
                                                                                    next_input_op.data.cpu().view(batch_size, 1),
                                                                                    1.0)).cuda()
                last_op = arr
            return outputs

    def beam_search_mode_1(self, data, w, max_time):
        """
        Implements beam search for different models.
        :param x: Input data
        :param w: beam width
        :param max_time: Maximum length till the program has to be generated
        :return all_beams: all beams to find out the indices of all the 
        """
        data, input_op = data

        # Beam, dictionary, with elements as list. Each element of list
        # containing index of the selected output and the corresponding
        # probability.
        data = data.permute(1, 0, 2, 3, 4)
        pytorch_version = torch.__version__[2]
        batch_size = data.size()[1]
        h = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
        # Last beams' data
        B = {0: {"input": input_op, "h": h}, 1: None}
        next_B = {}
        x_f = self.encoder(data[-1, :, 0:1, :, :])
        prev_output_prob = [Variable(torch.ones(batch_size, self.num_draws)).cuda()]
        all_beams = []
        all_inputs = []
        stopped_programs = np.zeros((batch_size, w), dtype=bool)
        for timestep in range(0, max_time):
            outputs = []
            for b in range(w):
                if not B[b]:
                    break
                input_op = B[b]["input"]

                h = B[b]["h"]
                input_op_rnn = self.relu(self.dense_input_op(input_op[:, 0, :]))
                input_op_rnn = input_op_rnn.view(1, batch_size,
                                                 self.input_op_sz)
                input = torch.cat((x_f, input_op_rnn), 2)
                h, _ = self.rnn(input, h)
                hd = self.relu(self.dense_fc_1(self.drop(h[0])))
                dense_output = self.dense_output(self.drop(hd))
                output = self.logsoftmax(dense_output)
                # Element wise multiply by previous probabs
                if pytorch_version == "3":
                    output = torch.nn.Softmax(1)(output)
                elif pytorch_version == "1":
                    output = torch.nn.Softmax()(output)
                output = output * prev_output_prob[b]
                outputs.append(output)
                next_B[b] = {}
                next_B[b]["h"] = h

            if len(outputs) == 1:
                outputs = outputs[0]
            else:
                outputs = torch.cat(outputs, 1)

            next_beams_index = torch.topk(outputs, w, 1, sorted=True)[1]
            next_beams_prob = torch.topk(outputs, w, 1, sorted=True)[0]
            # print (next_beams_prob)
            current_beams = {"parent": next_beams_index.data.cpu().numpy() // (
                self.num_draws),
                             "index": next_beams_index % (self.num_draws)}
            # print (next_beams_index % (self.num_draws))
            next_beams_index %= (self.num_draws)
            all_beams.append(current_beams)

            # Update previous output probabilities
            temp = Variable(torch.zeros(batch_size, 1)).cuda()
            prev_output_prob = []
            for i in range(w):
                for index in range(batch_size):
                    temp[index, 0] = next_beams_prob[index, i]
                prev_output_prob.append(temp.repeat(1, self.num_draws))
            # hidden state for next step
            B = {}
            for i in range(w):
                B[i] = {}
                temp = Variable(torch.zeros(h.size())).cuda()
                for j in range(batch_size):
                    temp[0, j, :] = next_B[current_beams["parent"][j, i]]["h"][0, j, :]
                B[i]["h"] = temp

            # one_hot for input to the next step
            for i in range(w):
                arr = Variable(torch.zeros(batch_size, self.num_draws + 1)
                               .scatter_(1, next_beams_index[:, i:i + 1].data.cpu(),
                                         1.0)).cuda()
                B[i]["input"] = arr.unsqueeze(1)
            all_inputs.append(B)

        return all_beams, next_beams_prob, all_inputs