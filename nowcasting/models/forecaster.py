from torch import nn
import torch
from nowcasting.utils import make_layers
from nowcasting.config import cfg
import logging

# Forecaster预测器的定义
# subnets属性为列表,包含网络使用激活函数及该层网络的维度
# rnns属性为列表,包含多层网络的实例
# rnns与subnets每个值一一对应，表示同一层网络
class Forecaster(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        # 网络层数
        self.blocks = len(subnets)
        # 将网络信息存储于一个Forecaster对象的属性中
        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            # 根据网络信息,通过make_layers函数构建网络
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))
    # 单层网络的前向传播
    # input为五维度:序列长度,一次训练所选取的训练数据数量,输入通道数,高度,宽度 5D S*B*I*H*W
    # subnet及rnn由初始化时使用的参数决定,不同的循环神经网络将复用同一个预测器前向传播过程
    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=cfg.HKO.BENCHMARK.OUT_LEN)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

        return input

    # 串联每层网络的前向传播
    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'),
                                      getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)))
        return input
