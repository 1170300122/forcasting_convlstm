from torch import nn
import torch
from nowcasting.utils import make_layers
from nowcasting.config import cfg
import logging

# Encoder编码器的定义
# subnets属性为列表,包含网络使用激活函数及该层网络的维度
# rnns属性为列表,包含多层网络的实例
# rnns与subnets每个值一一对应，表示同一层网络
class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        # 断言检测rnns与subnets的层数是否相同,这波很细节
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)
        # 将网络信息存储于一个Encoder对象的属性中
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)
    # 单层网络的前向传播
    # input为五维度:序列长度,一次训练所选取的训练数据数量,输入通道数,高度,宽度 5D S*B*I*H*W
    # subnet及rnn由初始化时使用的参数决定,不同的循环神经网络将复用同一个编码器的前向传播过程
    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        # hidden = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # cell = torch.zeros((batch_size, rnn._cell._hidden_size, input.size(3), input.size(4))).to(cfg.GLOBAL.DEVICE)
        # state = (hidden, cell)
        outputs_stage, state_stage = rnn(input, None)

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    # 串联每一层的前向传播过程,构成总的前向传播过程
    def forward(self, input):
        hidden_states = []
        logging.debug(input.size())
        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)

